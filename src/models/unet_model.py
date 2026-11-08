"""U-Net spatial prospectivity model (XGBoost challenger).

Spatial context advantage
--------------------------
Mineral systems leave multi-scale alteration halos that are invisible to
pixel-wise classifiers.  A U-Net with a 64 × 64 pixel receptive field at
50 m resolution spans a **3.2 km window**, large enough to capture:

* Distal potassic → proximal phyllic → argillic → propylitic zonation
  sequences typical of porphyry Cu–Au systems.
* Structural grain (fault corridor widths, damage-zone extents).
* Circular gravity / magnetic footprints of buried intrusions.

The encoder–decoder skip architecture preserves fine-grained positional
accuracy (essential for drill-target polygons) while integrating regional
context at deeper layers.

When to prefer U-Net over XGBoost
-----------------------------------
Adopt the U-Net challenger only when **both** conditions are satisfied
(evaluated by ``evaluation.metrics.compare_models``):

1. PR-AUC improvement > 5 percentage points on held-out spatial folds.
2. Wilcoxon signed-rank p-value < 0.05 (statistically significant uplift).

If either condition fails, retain the interpretable XGBoost baseline.

Epistemic uncertainty via MC-Dropout
--------------------------------------
At inference time the model is kept in training mode so that ``Dropout2d``
layers remain active.  Running ``n_passes`` (default 50) stochastic forward
passes and computing their variance yields a **Monte-Carlo Dropout** estimate
of epistemic uncertainty (Gal & Ghahramani, 2016).  High-variance regions
correspond to pixels where the network has seen few similar training examples
or conflicting spatial patterns — the primary input to active-learning loops
for new survey acquisition.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion import precision_weighted_log_odds_fusion

logger = logging.getLogger(__name__)


def get_best_device() -> torch.device:
    """Select the best available compute device (CUDA -> CPU / MPS).

    Allows override via SIFT_DEVICE environment variable ('cpu', 'cuda', 'mps').
    Defaults to CPU on macOS to eliminate Apple Silicon Metal command buffer
    synchronization stalls during micro-batch on-the-fly patch inference.
    """
    dev_env = os.environ.get("SIFT_DEVICE", "").strip().lower()
    if dev_env in ("cpu", "cuda", "mps"):
        return torch.device(dev_env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Building-block modules
# ---------------------------------------------------------------------------


class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm2d → ReLU blocks.

    This is the atomic building block of the U-Net architecture.  Using two
    3 × 3 convolutions per resolution level (instead of one 5 × 5) reduces
    parameter count while maintaining the same receptive field and adding an
    extra non-linearity, which improves gradient flow during training.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        mid_channels (int | None): Optional intermediate channel count;
            defaults to ``out_channels``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the double convolution block.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, C_out, H, W)``.
        """
        # Guard against BatchNorm2d ValueError when training on a single 1x1 spatial element
        if self.training and x.shape[0] == 1 and x.shape[2] == 1 and x.shape[3] == 1:
            out = x
            for layer in self.double_conv:
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
                    out = layer(out)
                    layer.train()
                else:
                    out = layer(out)
            return out
        return self.double_conv(x)


class Down(nn.Module):
    """Encoder step: MaxPool2d spatial downsampling followed by DoubleConv.

    Halves the spatial resolution (H, W → H/2, W/2) while doubling the
    channel depth, progressively compressing spatial information into a
    higher-dimensional semantic representation.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample and convolve.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            torch.Tensor: Downsampled output of shape ``(B, C_out, H/2, W/2)``.
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Decoder step: bilinear upsampling + skip-connection concatenation + DoubleConv.

    Restores spatial resolution from the bottleneck back to the input size.
    The skip connection from the corresponding encoder level is concatenated
    channel-wise before the DoubleConv, allowing the decoder to recover
    fine-grained structural detail lost during downsampling — critical for
    accurate drill-target polygon delineation.

    Bilinear interpolation is preferred over transposed convolution because it
    avoids the checkerboard artefacts that degrade spatial coherence of
    prospectivity maps.

    Args:
        in_channels (int): Combined channels after concatenation with skip
            (encoder_channels + decoder_channels).
        out_channels (int): Desired output channels after DoubleConv.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Upsample ``x1`` and concatenate with skip-connection ``x2``.

        Args:
            x1 (torch.Tensor): Decoder tensor to upsample,
                shape ``(B, C1, H, W)``.
            x2 (torch.Tensor): Skip-connection tensor from encoder,
                shape ``(B, C2, H*2, W*2)``.

        Returns:
            torch.Tensor: Merged output of shape
                ``(B, out_channels, H*2, W*2)``.
        """
        x1 = self.up(x1)

        # Pad x1 to match x2 spatial dims if they differ (due to odd sizes)
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1,
            [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
        )

        # Concatenate skip connection along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full U-Net architecture
# ---------------------------------------------------------------------------


class ProspectivityUNet(nn.Module):
    """Fully-convolutional U-Net for pixel-wise prospectivity scoring.

    Architecture (default):
    ::

        Input (in_channels, 64, 64)
            └─ DoubleConv → 64 ch            [enc1]
                └─ Down → 128 ch             [enc2]
                    └─ Down → 256 ch         [enc3]
                        └─ Down → 512 ch     [enc4]
                            └─ Down+Dropout → 1024 ch  [bottleneck]
                        ↑─ Up → 512 ch       [dec4]
                    ↑─ Up → 256 ch           [dec3]
                ↑─ Up → 128 ch               [dec2]
            ↑─ Up → 64 ch                    [dec1]
        Output Conv2d(64→1) + Sigmoid → (1, 64, 64) prospectivity map

    The final Sigmoid squashes output to [0, 1], interpretable as pixel-level
    probability of hosting mineralisation.

    MC-Dropout uncertainty:
        ``Dropout2d(p=dropout_p)`` is applied in the bottleneck.  At inference
        time, calling ``model.train()`` keeps dropout active, enabling
        Monte-Carlo sampling over the weight-dropout posterior.

    Args:
        in_channels (int): Number of input raster bands / features.
        dropout_p (float): Spatial dropout probability applied at the
            bottleneck.  Default 0.3 follows the empirical recommendation of
            Gal & Ghahramani (2016) for MC-Dropout uncertainty estimation.
    """

    def __init__(self, in_channels: int, dropout_p: float = 0.3) -> None:
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.inc = DoubleConv(in_channels, 64)   # 64 × 64 → 64 ch
        self.down1 = Down(64, 128)               # 32 × 32 → 128 ch
        self.down2 = Down(128, 256)              # 16 × 16 → 256 ch
        self.down3 = Down(256, 512)              # 8 × 8   → 512 ch

        # ── Bottleneck (with MC-Dropout) ──────────────────────────────────────
        self.down4 = Down(512, 1024)             # 4 × 4   → 1024 ch
        self.bottleneck_drop = nn.Dropout2d(p=dropout_p)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)

        # ── Final classification head ─────────────────────────────────────────
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """U-Net forward pass with skip connections.

        Args:
            x (torch.Tensor): Input batch of shape
                ``(B, in_channels, H, W)``.

        Returns:
            torch.Tensor: Prospectivity map of shape ``(B, 1, H, W)``
                with values in [0, 1].
        """
        # Encoder
        x1 = self.inc(x)        # (B, 64,   H,    W)
        x2 = self.down1(x1)     # (B, 128,  H/2,  W/2)
        x3 = self.down2(x2)     # (B, 256,  H/4,  W/4)
        x4 = self.down3(x3)     # (B, 512,  H/8,  W/8)

        # Bottleneck
        x5 = self.down4(x4)           # (B, 1024, H/16, W/16)
        x5 = self.bottleneck_drop(x5) # MC-Dropout applied here

        # Decoder with skip connections
        x = self.up1(x5, x4)    # (B, 512,  H/8,  W/8)
        x = self.up2(x, x3)     # (B, 256,  H/4,  W/4)
        x = self.up3(x, x2)     # (B, 128,  H/2,  W/2)
        x = self.up4(x, x1)     # (B, 64,   H,    W)

        # Classification head
        logits = self.outc(x)   # (B, 1,    H,    W)
        return self.sigmoid(logits)


# ---------------------------------------------------------------------------
# Task 2.1: District-Scale DAE-U-Net (250-500 m) & Unsupervised Reconstruction
# ---------------------------------------------------------------------------


class SpatialAttention(nn.Module):
    """Spatial attention gate to weight multi-scale decoder features."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class ASPPBottleneck(nn.Module):
    """Resolution-adaptive 4-branch Atrous Spatial Pyramid Pooling (ASPP) bottleneck.

    Applies parallel dilations corresponding to physical radii [0.5, 1.5, 3.0, 6.0] km:
        dilation = max(1, round(radius_m / (16.0 * px_size_m)))
    """

    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 1024,
        px_size_m: float = 250.0,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        radii_m = [500.0, 1500.0, 3000.0, 6000.0]
        bottleneck_cell_m = 16.0 * float(px_size_m)
        dilations = [max(1, int(round(r / bottleneck_cell_m))) for r in radii_m]
        d1, d2, d3, d4 = dilations[0], dilations[1], dilations[2], dilations[3]
        mid = out_channels // 4

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=mid),
            nn.GELU(),
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=d1, dilation=d1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=mid),
            nn.GELU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=d2, dilation=d2, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=mid),
            nn.GELU(),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=d3, dilation=d3, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=mid),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid * 4, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.GELU(),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out0 = self.b0(x)
        out1 = self.b1(x)
        out2 = self.b2(x)
        out3 = self.b3(x)
        out = torch.cat([out0, out1, out2, out3], dim=1)
        return self.project(out)


class DistrictDAEUNet(nn.Module):
    """District-scale Deep Autoencoder U-Net (DAE-U-Net) with K=4 heads.

    Enforces Task 2.1 & v4.0 upgrades (D4, D6, Bug 8):
      1. Evaluated on decimated 250 m or 500 m spatial grids to span 16-32 km
         district-scale plumbing within a 16 GB RAM budget.
      2. 15% random-drop channel augmentation during training.
      3. Resolution-adaptive 4-branch ASPP bottleneck ([0.5, 1.5, 3.0, 6.0] km).
      4. Unsupervised Deep Autoencoder (DAE) Reconstruction Decoder Head connected
         strictly to the bottleneck WITHOUT skip connections (preventing identity copying).
      5. K=4 spatial attention heads with spatial attention gating and output bias
         initialization to logit(pi_k).
      6. Generates pixel-wise reconstruction error (dae_recon_err) and spatial context
         embeddings (unet_ctx_*).
    """

    def __init__(
        self,
        in_channels: int = 10,
        n_classes: int = 4,
        dropout_p: float = 0.3,
        channel_drop_p: float = 0.15,
        px_size_m: float = 250.0,
        pi_p: Union[float, Sequence[float]] = 0.05,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.channel_drop_p = channel_drop_p
        self.px_size_m = float(px_size_m)
        self.pi_p = pi_p

        # ── Encoder ──────────────────────────────────────────────────────────
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # ── Bottleneck with Resolution-Adaptive 4-Branch ASPP (D6, Bug 8) ─────
        self.down4 = Down(512, 1024)
        self.aspp = ASPPBottleneck(1024, 1024, px_size_m=self.px_size_m, dropout_p=dropout_p)
        self.bottleneck_drop = nn.Dropout2d(p=dropout_p)

        # ── Supervised Decoder (K=4 heads with skip connections & spatial attention) ──
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)
        self.sa = SpatialAttention(kernel_size=7)
        self.out_heads = nn.Conv2d(64, n_classes, kernel_size=1)

        # Output bias initialization matching logit(pi_k) (D4)
        if isinstance(pi_p, (int, float)):
            p_val = float(np.clip(float(pi_p), 1e-4, 1.0 - 1e-4))
            init_b = float(np.log(p_val / (1.0 - p_val)))
            nn.init.constant_(self.out_heads.bias, init_b)
        else:
            p_seq = list(pi_p)
            for k in range(n_classes):
                pk = float(p_seq[k]) if k < len(p_seq) else float(p_seq[0])
                pk = float(np.clip(pk, 1e-4, 1.0 - 1e-4))
                init_b = float(np.log(pk / (1.0 - pk)))
                nn.init.constant_(self.out_heads.bias[k], init_b)
        self.sigmoid = nn.Sigmoid()

        # ── Unsupervised DAE Reconstruction Decoder (Bottleneck ONLY, NO skips) ─
        self.dae_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConv(1024, 512),
        )
        self.dae_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConv(512, 256),
        )
        self.dae_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConv(256, 128),
        )
        self.dae_up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConv(128, 64),
        )
        self.dae_recon = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with channel augmentation, K=4 heads, and bottleneck DAE reconstruction.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Tuple (preds, recon, recon_err, ctx_embeddings) where:
              - preds: (B, n_classes, H, W) prospectivity probabilities in [0, 1]
              - recon: (B, in_channels, H, W) reconstructed features
              - recon_err: (B, 1, H, W) pixel-wise squared reconstruction error
              - ctx_embeddings: (B, 64, H, W) spatial context feature map
        """
        x_orig = x

        # 15% random-drop channel augmentation during training
        if self.training and self.channel_drop_p > 0:
            mask = (torch.rand(x.size(0), x.size(1), 1, 1, device=x.device) >= self.channel_drop_p).float()
            x = x * mask / (1.0 - self.channel_drop_p)

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck with ASPP multi-scale receptive field (D6, Bug 8)
        x5 = self.aspp(self.down4(x4))

        # Supervised Decoder (K=4 heads with skip connections & spatial attention)
        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)
        d4_att = self.sa(d4)
        preds = self.sigmoid(self.out_heads(d4_att))

        # Unsupervised DAE Reconstruction Decoder (Bottleneck ONLY, strictly NO skips)
        r1 = self.dae_up1(x5)
        r2 = self.dae_up2(r1)
        r3 = self.dae_up3(r2)
        r4 = self.dae_up4(r3)
        recon = self.dae_recon(r4)

        # Pixel-wise reconstruction error across channels
        recon_err = torch.mean((x_orig - recon) ** 2, dim=1, keepdim=True)

        return preds, recon, recon_err, d4_att


def decimate_spatial_grid(
    grid: np.ndarray,
    target_res_m: float = 250.0,
    native_res_m: float = 50.0,
    order: int = 1,
) -> np.ndarray:
    """Decimate spatial feature grid to 250 m or 500 m prior to neural evaluation.

    Args:
        grid: 2-D, 3-D, or 4-D numpy array.
        target_res_m: Decimated resolution in metres (e.g. 250.0 or 500.0).
        native_res_m: Native resolution in metres (e.g. 50.0).
        order: Interpolation order (1 = bilinear for continuous features,
               0 = nearest-neighbor for categorical fold IDs / binary labels).

    Returns:
        Decimated grid array.
    """
    scale = float(native_res_m) / float(target_res_m)
    if np.isclose(scale, 1.0):
        return grid

    arr = np.asarray(grid, dtype=np.int32 if order == 0 and np.issubdtype(np.asarray(grid).dtype, np.integer) else np.float32)
    if arr.ndim == 2:
        out = zoom(arr, zoom=scale, order=order, mode="nearest")
    elif arr.ndim == 3:
        out = zoom(arr, zoom=(1.0, scale, scale), order=order, mode="nearest")
    elif arr.ndim == 4:
        out = zoom(arr, zoom=(1.0, 1.0, scale, scale), order=order, mode="nearest")
    else:
        raise ValueError(f"Unsupported grid dimension {arr.ndim}")
    if order == 0 and np.issubdtype(np.asarray(grid).dtype, np.integer):
        assert set(np.unique(out)).issubset(set(np.unique(grid))), (
            "FATAL: Interpolated fold IDs detected in decimated categorical grid!"
        )
    return out


def project_spatial_grid(
    coarse_grid: np.ndarray,
    target_shape: tuple[int, int],
    order: int = 1,
) -> np.ndarray:
    """Project coarse district-scale embeddings/reconstruction error back to 50 m grid via scipy.ndimage.zoom.

    Args:
        coarse_grid: Coarse grid of shape (H_c, W_c) or (C, H_c, W_c).
        target_shape: (H_target, W_target) of the native high-resolution grid.
        order: Spline interpolation order (1 = bilinear, default).

    Returns:
        Projected array matching target_shape.
    """
    arr = np.asarray(coarse_grid, dtype=np.float32)
    H_t, W_t = target_shape
    if arr.ndim == 2:
        H_c, W_c = arr.shape
        zoom_factors = (H_t / H_c, W_t / W_c)
        proj = zoom(arr, zoom=zoom_factors, order=order)
        return proj[:H_t, :W_t]
    elif arr.ndim == 3:
        C, H_c, W_c = arr.shape
        zoom_factors = (1.0, H_t / H_c, W_t / W_c)
        proj = zoom(arr, zoom=zoom_factors, order=order)
        return proj[:, :H_t, :W_t]
    else:
        raise ValueError(f"Unsupported coarse_grid dimension {arr.ndim}")



# ---------------------------------------------------------------------------
# Pass 3.2: Non-Negative PU (nnPU) Risk Loss (Kiryo et al., NeurIPS 2017)
# ---------------------------------------------------------------------------


class nnPULoss(nn.Module):
    """Non-Negative Positive-Unlabeled (nnPU) Risk Estimator with Propensity Weights.

    Implements the unbiased PU risk estimator with non-negativity clamping and
    gradient reversal (Kiryo et al., 2017) augmented with Pass 2 inverse
    propensity weights $w_i = 1 / e(x_i)$:

    $$\\hat{R}_{\\text{nnPU}}(g) = \\pi_p \\hat{R}_p^+(g) + \\max\\left(0,\\; \\hat{R}_u^-(g) - \\pi_p \\hat{R}_p^-(g)\\right)$$

    To satisfy **Acceptance Criterion 3 (nnPU Non-Negativity)**:
      1. The forward risk value clamps the empirical unlabeled negative risk term
         $r_{\\text{neg}} = \\hat{R}_u^-(g) - \\pi_p \\hat{R}_p^-(g)$ at $0$ whenever
         $r_{\\text{neg}} < -\\beta$ (default $\\beta = 0.0$).
      2. When $r_{\\text{neg}} < -\\beta$, ``last_clamped`` is set to ``True`` and
         if ``gamma > 0`` a reversed gradient term $-\\gamma \\cdot r_{\\text{neg}}$
         is used for the backward pass (via straight-through / detached value
         substitution so the forward scalar remains strictly $\\pi_p \\hat{R}_p^+(g)$
         while the backward gradient ascends $r_{\\text{neg}}$ to actively correct
         overfitting on unlabeled positives).

    Args:
        pi_p: Class prior $\\pi_p = P(Y = +1) \\in (0, 1)$. Defaults to ``0.05``.
        gamma: Gradient reversal strength when $r_{\\text{neg}} < -\\beta$.
            Defaults to ``1.0`` (set to ``0.0`` for pure zero-gradient clamping).
        beta: Tolerance threshold for negative risk. Defaults to ``0.0``.
    """

    def __init__(
        self,
        pi_p: Union[float, Sequence[float], torch.Tensor] = 0.05,
        gamma: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(pi_p, (int, float)):
            if not (0.0 < float(pi_p) < 1.0):
                raise ValueError(f"pi_p must be in (0, 1), got {pi_p}")
            self.pi_p: float = float(pi_p)
            self.pi_p_list: Optional[list[float]] = None
        else:
            if isinstance(pi_p, torch.Tensor):
                p_list = [float(x) for x in pi_p.detach().cpu().numpy().ravel()]
            else:
                p_list = [float(x) for x in pi_p]
            for p_val in p_list:
                if not (0.0 < p_val < 1.0):
                    raise ValueError(f"All per-head priors must be in (0, 1), got {p_val}")
            self.pi_p: float = float(p_list[0]) if len(p_list) > 0 else 0.05
            self.pi_p_list: Optional[list[float]] = p_list

        self.gamma: float = float(gamma)
        self.beta: float = float(beta)
        self.last_clamped: bool = False
        self.last_unlabeled_risk: float = 0.0

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute nnPU risk loss independently per class channel k.

        Args:
            preds: Predicted probabilities in $[0, 1]$, shape ``(B, ...)`` or ``(B, K, ...)``.
            targets: Binary/PU labels where ``1`` = positive ($P$) and ``0`` or
                ``-1`` = unlabeled ($U$), broadcastable to ``preds``.
            sample_weights: Optional inverse propensity weights $w_i = 1 / e(x_i)$,
                broadcastable to ``preds``.

        Returns:
            Scalar nnPU loss tensor averaged across class heads.
        """
        eps = 1e-7
        p = torch.clamp(preds, eps, 1.0 - eps)
        t = targets.expand_as(p)

        if sample_weights is not None:
            w = sample_weights.expand_as(p)
        else:
            w = torch.ones_like(p)

        if p.ndim == 1:
            n_heads = 1
            p_heads = [p]
            t_heads = [t]
            w_heads = [w]
        elif p.ndim >= 2:
            n_heads = p.shape[1]
            p_heads = [p[:, k] for k in range(n_heads)]
            t_heads = [t[:, k] for k in range(n_heads)]
            w_heads = [w[:, k] for k in range(n_heads)]

        head_losses: list[torch.Tensor] = []
        clamped_heads: list[bool] = []
        r_neg_heads: list[float] = []

        for k in range(n_heads):
            pk = p_heads[k]
            tk = t_heads[k]
            wk = w_heads[k]

            pos_mask_k = (tk > 0.5).float()
            unl_mask_k = 1.0 - pos_mask_k

            has_pos_k = (torch.sum(pos_mask_k) > 0)
            if not has_pos_k:
                # Bug 3: Mask zero-positive heads from loss updates
                continue

            pi_k = (
                self.pi_p_list[k]
                if (self.pi_p_list is not None and k < len(self.pi_p_list))
                else self.pi_p
            )

            loss_pos_k = -torch.log(pk)
            loss_neg_k = -torch.log(1.0 - pk)

            # Hájek self-normalized inverse-propensity weights (D4)
            pos_denom_k = torch.clamp(torch.sum(pos_mask_k), min=1.0)
            unl_denom_k = torch.clamp(torch.sum(unl_mask_k), min=1.0)

            pos_w_sum = torch.sum(pos_mask_k * wk)
            mean_pos_w = torch.clamp(pos_w_sum / pos_denom_k, min=1e-6)
            w_pos_hajek = wk / mean_pos_w

            r_p_plus_k = torch.sum(pos_mask_k * w_pos_hajek * loss_pos_k) / pos_denom_k
            r_p_minus_k = torch.sum(pos_mask_k * w_pos_hajek * loss_neg_k) / pos_denom_k
            r_u_minus_k = torch.sum(unl_mask_k * loss_neg_k) / unl_denom_k

            r_neg_k = r_u_minus_k - pi_k * r_p_minus_k
            r_neg_val = float(r_neg_k.detach().cpu().item())
            r_neg_heads.append(r_neg_val)

            if r_neg_val < -self.beta:
                clamped_heads.append(True)
                clamped_forward_k = pi_k * r_p_plus_k
                if self.gamma > 0.0:
                    reversed_grad_term_k = -self.gamma * r_neg_k
                    loss_k = clamped_forward_k.detach() + (
                        reversed_grad_term_k - reversed_grad_term_k.detach()
                    )
                else:
                    loss_k = clamped_forward_k
            else:
                clamped_heads.append(False)
                loss_k = pi_k * r_p_plus_k + r_neg_k

            head_losses.append(loss_k)

        self.last_clamped = any(clamped_heads)
        self.last_unlabeled_risk = float(np.mean(r_neg_heads)) if r_neg_heads else 0.0

        if len(head_losses) == 0:
            return torch.zeros((), device=p.device, dtype=p.dtype, requires_grad=True)
        if len(head_losses) == 1:
            return head_losses[0]
        return torch.mean(torch.stack(head_losses))



# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------


class MPMUNet:
    """High-level wrapper for training and inference with ProspectivityUNet.

    Handles:
    * Dataset construction from numpy window arrays.
    * Weighted binary cross-entropy for class imbalance.
    * AdamW optimisation with Cosine Annealing learning-rate schedule.
    * MC-Dropout inference for mean score + epistemic uncertainty maps.
    * Model persistence (state_dict checkpointing).

    Args:
        config (dict): Pipeline config; must contain the key path
            ``config["models"]["unet"]`` with at minimum:
            ``window_px`` (int), ``in_channels`` (int),
            ``dropout_p`` (float, default 0.3),
            ``batch_size`` (int, default 32).

    Attributes:
        window_px (int): Spatial window size in pixels (square).
        in_channels (int): Number of raster input bands.
        dropout_p (float): Bottleneck dropout probability.
        device (torch.device): CPU or CUDA device used for training.
        model_ (ProspectivityUNet | None): Fitted model (set by ``fit``).
    """

    def __init__(self, config: dict) -> None:
        models_cfg = config.get("models", {})
        unet_cfg: dict = (
            models_cfg.get("unet")
            or models_cfg.get("challenger")
            or {}
        )

        self.window_px: int = int(
            unet_cfg.get("window_px", unet_cfg.get("input_window_px", 64))
        )
        self.in_channels: int = int(unet_cfg.get("in_channels", 10))
        self.dropout_p: float = float(unet_cfg.get("dropout_p", 0.3))
        self.batch_size: int = int(unet_cfg.get("batch_size", 32))
        self.pi_p: float = float(unet_cfg.get("pi_p", 0.05))
        self.use_nnpu: bool = bool(unet_cfg.get("use_nnpu", True))

        self.device: torch.device = get_best_device()
        self.model_: Optional[ProspectivityUNet] = None
        self.criterion_: Optional[nnPULoss] = None

        logger.info(
            "MPMUNet initialised | window_px=%d | in_channels=%d | device=%s",
            self.window_px,
            self.in_channels,
            self.device,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_dataset(
        self,
        X_windows: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> TensorDataset:
        """Wrap numpy arrays as a PyTorch TensorDataset.

        Args:
            X_windows (np.ndarray): Window stack of shape
                ``(N, in_channels, window_px, window_px)``.
            y (np.ndarray): Binary labels of shape ``(N,)``.
            sample_weight (np.ndarray | None): Optional inverse propensity
                weights of shape ``(N,)``.

        Returns:
            torch.utils.data.TensorDataset: Dataset usable by DataLoader.
        """
        X_t = torch.tensor(X_windows, dtype=torch.float32)
        # Labels need shape (N, 1, 1, 1) to broadcast against (N, 1, H, W) output.
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1, 1, 1)
        if sample_weight is not None:
            w_t = torch.tensor(sample_weight, dtype=torch.float32).view(-1, 1, 1, 1)
        else:
            w_t = torch.ones_like(y_t)
        return TensorDataset(X_t, y_t, w_t)

    def _compute_pos_weight(self, y: np.ndarray) -> torch.Tensor:
        """Compute BCELoss pos_weight from label array.

        Args:
            y (np.ndarray): Binary label array.

        Returns:
            torch.Tensor: Scalar tensor = n_negative / n_positive.
        """
        n_pos = int((y > 0.5).sum())
        n_neg = len(y) - n_pos
        if n_pos == 0:
            raise ValueError("Training set contains no positive samples.")
        spw = max(n_neg / n_pos, 1.0)
        return torch.tensor([spw], dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_windows: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 50,
        lr: float = 1e-4,
        sample_weight: Optional[np.ndarray] = None,
        pi_p: Optional[float] = None,
        use_nnpu: Optional[bool] = None,
    ) -> "MPMUNet":
        """Train ProspectivityUNet with Non-Negative PU (nnPU) risk loss + AdamW.

        Training recipe:
        * **Loss**: Non-Negative PU (nnPU) risk estimator :class:`nnPULoss`
          incorporating class prior $\\pi_p$ and Pass 2 inverse propensity
          weights $w_i = 1 / e(x_i)$ (or weighted BCE fallback if ``use_nnpu=False``).
        * **Optimiser**: AdamW with weight decay 1e-4 to regularise the ~8 M
          parameter network on limited geological training sets.
        * **Scheduler**: Cosine Annealing from ``lr`` to ``lr/100`` over
          ``epochs``.

        Args:
            X_windows (np.ndarray): Window stack, shape
                ``(N, in_channels, window_px, window_px)``.
            y (np.ndarray): Binary/PU labels, shape ``(N,)``; 1 = deposit, 0/-1 = U.
            val_data (tuple | None): Optional ``(X_val, y_val)`` with same
                shapes as training arrays for per-epoch validation loss logging.
            epochs (int): Number of full training epochs.  Default 50.
            lr (float): Initial learning rate for AdamW.  Default 1e-4.
            sample_weight (np.ndarray | None): Optional Pass 2 inverse propensity
                weights $w_i$ of shape ``(N,)``.
            pi_p (float | None): Optional class prior override $\\pi_p \\in (0, 1)$.
            use_nnpu (bool | None): Whether to use :class:`nnPULoss` (default True).

        Returns:
            MPMUNet: self, to allow method chaining.
        """
        # Build model
        self.model_ = ProspectivityUNet(
            in_channels=self.in_channels, dropout_p=self.dropout_p
        ).to(self.device)

        active_nnpu = self.use_nnpu if use_nnpu is None else bool(use_nnpu)
        active_pi_p = self.pi_p if pi_p is None else pi_p

        # Positive class weight for imbalanced BCE fallback
        pos_weight = self._compute_pos_weight(np.asarray(y))
        nnpu_criterion = nnPULoss(pi_p=active_pi_p).to(self.device)
        self.criterion_ = nnpu_criterion

        def weighted_bce(
            pred: torch.Tensor,
            target: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            pw = pos_weight.expand_as(target)
            w = weights.expand_as(target) if weights is not None else 1.0
            return torch.mean(
                w
                * -(
                    pw * target * torch.log(pred + 1e-8)
                    + (1 - target) * torch.log(1 - pred + 1e-8)
                )
            )

        optimiser = torch.optim.AdamW(
            self.model_.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=max(epochs, 1), eta_min=lr / 100
        )

        # Build data loaders
        train_ds = self._build_dataset(X_windows, y, sample_weight=sample_weight)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        val_loader: Optional[DataLoader] = None
        if val_data is not None:
            val_ds = self._build_dataset(val_data[0], val_data[1])
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0
            )

        # Training loop
        for epoch in range(1, epochs + 1):
            self.model_.train()
            train_losses: list[float] = []

            for X_batch, y_batch, w_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                w_batch = w_batch.to(self.device)

                optimiser.zero_grad()
                preds = self.model_(X_batch)  # (B, 1, H, W)
                # Extract center pixel (B, 1, 1, 1) when targets are scalar window labels
                if y_batch.shape[-2:] == (1, 1) and preds.shape[-2:] != (1, 1):
                    hc, wc = preds.shape[-2] // 2, preds.shape[-1] // 2
                    preds_target = preds[:, :, hc : hc + 1, wc : wc + 1]
                else:
                    preds_target = preds

                if active_nnpu:
                    loss = nnpu_criterion(preds_target, y_batch, sample_weights=w_batch)
                else:
                    y_expanded = (y_batch > 0.5).float().expand_as(preds_target)
                    loss = weighted_bce(preds_target, y_expanded, weights=w_batch)

                loss.backward()
                optimiser.step()
                train_losses.append(loss.item())

            scheduler.step()
            avg_train = np.mean(train_losses)

            # Optional validation loss
            if val_loader is not None:
                self.model_.eval()
                val_losses: list[float] = []
                with torch.no_grad():
                    for X_val_b, y_val_b, w_val_b in val_loader:
                        X_val_b = X_val_b.to(self.device)
                        y_val_b = y_val_b.to(self.device)
                        w_val_b = w_val_b.to(self.device)
                        preds_val = self.model_(X_val_b)
                        if y_val_b.shape[-2:] == (1, 1) and preds_val.shape[-2:] != (1, 1):
                            hc, wc = preds_val.shape[-2] // 2, preds_val.shape[-1] // 2
                            preds_val_target = preds_val[:, :, hc : hc + 1, wc : wc + 1]
                        else:
                            preds_val_target = preds_val
                        if active_nnpu:
                            val_loss = nnpu_criterion(
                                preds_val_target, y_val_b, sample_weights=w_val_b
                            )
                        else:
                            y_val_expanded = (y_val_b > 0.5).float().expand_as(preds_val_target)
                            val_loss = weighted_bce(
                                preds_val_target, y_val_expanded, weights=w_val_b
                            )
                        val_losses.append(val_loss.item())
                avg_val = np.mean(val_losses)
                logger.info(
                    "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | lr=%.2e",
                    epoch, epochs, avg_train, avg_val,
                    scheduler.get_last_lr()[0],
                )
            else:
                logger.info(
                    "Epoch %3d/%d | train_loss=%.4f | lr=%.2e",
                    epoch, epochs, avg_train,
                    scheduler.get_last_lr()[0],
                )

        logger.info("MPMUNet training complete.")
        return self

    def predict_proba_mc(
        self,
        X_windows: np.ndarray,
        n_passes: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Monte-Carlo Dropout inference returning mean score + epistemic variance.

        Enables dropout at inference time by keeping the model in training mode
        (``model.train()``), then runs ``n_passes`` stochastic forward passes.
        The mean and variance across passes provide:

        * **Mean score** — best estimate of prospectivity probability,
          lower-variance than a single forward pass.
        * **Variance** — epistemic uncertainty; high where the model's
          dropout-perturbed weight samples disagree.  Use as an overlay on
          the prospectivity map to flag data gaps for new survey design.

        Args:
            X_windows (np.ndarray): Window stack, shape
                ``(N, in_channels, window_px, window_px)``.
            n_passes (int): Number of stochastic forward passes.  50 is
                sufficient for stable variance estimates; increase to 100 for
                publication-quality uncertainty maps.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                * ``mean_score`` — shape ``(N,)``, values in [0, 1].
                * ``variance``   — shape ``(N,)``, epistemic uncertainty.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict_proba_mc().")

        X_t = torch.tensor(X_windows, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        # Set model (including BatchNorm2d) to eval(), then enable only Dropout layers
        self.model_.eval()
        for module in self.model_.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()

        all_pass_scores: list[np.ndarray] = []

        for _ in range(n_passes):
            pass_scores: list[np.ndarray] = []
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                with torch.no_grad():
                    preds = self.model_(X_batch)  # (B, 1, H, W)
                hc, wc = preds.shape[-2] // 2, preds.shape[-1] // 2
                score = preds[:, 0, hc, wc].cpu().numpy()  # (B,)
                pass_scores.append(score)
            all_pass_scores.append(np.concatenate(pass_scores, axis=0))  # (N,)

        # Stack across passes: shape (n_passes, N)
        stacked = np.stack(all_pass_scores, axis=0)
        mean_score: np.ndarray = stacked.mean(axis=0)   # (N,)
        variance: np.ndarray = stacked.var(axis=0)      # (N,)

        return mean_score, variance

    def save(self, path: str) -> None:
        """Save model state_dict to disk.

        Saves only the model weights (not the full object), following the
        PyTorch convention for portable checkpoints that are version-safe
        across Python environments.

        Args:
            path (str): Destination ``.pt`` or ``.pth`` file path.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before save().")
        checkpoint = {
            "state_dict": self.model_.state_dict(),
            "in_channels": self.in_channels,
            "dropout_p": self.dropout_p,
            "window_px": self.window_px,
        }
        torch.save(checkpoint, path)
        logger.info("MPMUNet state_dict saved → %s", path)

    def load(self, path: str) -> "MPMUNet":
        """Load model weights from a checkpoint created by ``save()``.

        Reconstructs the ``ProspectivityUNet`` architecture from the saved
        metadata and loads the state_dict weights.

        Args:
            path (str): Path to the ``.pt`` / ``.pth`` checkpoint file.

        Returns:
            MPMUNet: self, to allow method chaining.
        """
        checkpoint: dict = torch.load(path, map_location=self.device)
        self.in_channels = checkpoint.get("in_channels", self.in_channels)
        self.dropout_p = checkpoint.get("dropout_p", self.dropout_p)
        self.window_px = checkpoint.get("window_px", self.window_px)

        self.model_ = ProspectivityUNet(
            in_channels=self.in_channels, dropout_p=self.dropout_p
        ).to(self.device)
        self.model_.load_state_dict(checkpoint["state_dict"])
        self.model_.eval()

        logger.info(
            "MPMUNet weights loaded ← %s | device=%s", path, self.device
        )
        return self


# ---------------------------------------------------------------------------
# High-Level Wrapper for District DAE-U-Net
# ---------------------------------------------------------------------------


class MPMDistrictDAEUNet:
    """High-level wrapper for training and inference with DistrictDAEUNet.

    Enforces Task 2.1:
      - Multi-scale district context (16-32 km) on decimated 250 m or 500 m grids.
      - Supervised K=4 deposit prospectivity heads with spatial attention.
      - Unsupervised bottleneck DAE reconstruction head for anomaly detection.
      - Cross-validation across 5 spatially buffered blocks.
      - Out-of-fold projection of unet_ctx_* and dae_recon_err back to 50 m grid.
    """

    def __init__(
        self,
        in_channels: int = 10,
        n_classes: int = 4,
        window_px: int = 32,
        batch_size: int = 16,
        dropout_p: float = 0.3,
        channel_drop_p: float = 0.15,
        recon_weight: float = 0.5,
        pi_p: Union[float, Sequence[float]] = 0.05,
        px_size_m: float = 250.0,
        subcanopy_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.window_px = window_px
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.channel_drop_p = channel_drop_p
        self.recon_weight = recon_weight
        self.pi_p = pi_p
        self.px_size_m = float(px_size_m)
        self.subcanopy_indices = [int(i) for i in subcanopy_indices] if subcanopy_indices is not None else None

        self.device = get_best_device()
        self.model: Optional[DistrictDAEUNet] = None
        self.criterion = nnPULoss(pi_p=self.pi_p).to(self.device)

    def fit(
        self,
        X_windows: np.ndarray,
        y_targets: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        epochs: int = 20,
        lr: float = 1e-4,
    ) -> MPMDistrictDAEUNet:
        """Fit DistrictDAEUNet with joint nnPU loss + DAE reconstruction loss."""
        self.model = DistrictDAEUNet(
            in_channels=self.in_channels,
            n_classes=self.n_classes,
            dropout_p=self.dropout_p,
            channel_drop_p=self.channel_drop_p,
            px_size_m=self.px_size_m,
            pi_p=self.pi_p,
        ).to(self.device)

        X_t = torch.tensor(X_windows, dtype=torch.float32)
        y_arr = np.asarray(y_targets, dtype=np.float32)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, np.newaxis]
        if y_arr.ndim == 2:
            y_arr = y_arr[:, :, np.newaxis, np.newaxis]
        y_t = torch.tensor(y_arr, dtype=torch.float32)

        if sample_weights is not None:
            w_arr = np.asarray(sample_weights, dtype=np.float32)
            if w_arr.ndim == 1:
                w_arr = w_arr[:, np.newaxis]
            if w_arr.ndim == 2:
                w_arr = w_arr[:, :, np.newaxis, np.newaxis]
            w_t = torch.tensor(w_arr, dtype=torch.float32)
        else:
            w_t = torch.ones_like(y_t)

        dataset = TensorDataset(X_t, y_t, w_t)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=(len(dataset) > self.batch_size),
        )

        optimiser = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(epochs):
            for xb, yb, wb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                wb = wb.to(self.device)

                optimiser.zero_grad()
                preds, recon, recon_err, ctx = self.model(xb)

                # nnPU loss across K=4 heads
                if yb.shape[-2:] == (1, 1) and preds.shape[-2:] != (1, 1):
                    hc, wc = preds.shape[-2] // 2, preds.shape[-1] // 2
                    preds_target = preds[:, :, hc : hc + 1, wc : wc + 1]
                else:
                    preds_target = preds

                loss_pu = self.criterion(preds_target, yb, sample_weights=wb)
                if self.subcanopy_indices is not None and len(self.subcanopy_indices) > 0:
                    valid_sub = [i for i in self.subcanopy_indices if i < self.in_channels]
                    if len(valid_sub) > 0:
                        loss_recon = nn.functional.mse_loss(recon[:, valid_sub], xb[:, valid_sub])
                    else:
                        loss_recon = nn.functional.mse_loss(recon, xb)
                else:
                    loss_recon = nn.functional.mse_loss(recon, xb)
                loss = loss_pu + self.recon_weight * loss_recon

                loss.backward()
                optimiser.step()

        return self

    def predict_features(
        self, X_windows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict K=4 prospectivity, DAE reconstruction error, and spatial context embeddings.

        Args:
            X_windows: (N, in_channels, H, W) window stack.

        Returns:
            Tuple (preds, recon_err, ctx_embeddings):
              - preds: (N, n_classes) center-pixel prospectivity probabilities
              - recon_err: (N,) center-pixel reconstruction error
              - ctx_embeddings: (N, 4) principal context components
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        self.model.eval()
        X_t = torch.tensor(X_windows, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=self.batch_size, shuffle=False)

        preds_list, err_list, ctx_list = [], [], []

        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                preds, recon, recon_err, ctx = self.model(xb)
                hc, wc = preds.shape[-2] // 2, preds.shape[-1] // 2

                p_center = preds[:, :, hc, wc].cpu().numpy()
                err_center = recon_err[:, 0, hc, wc].cpu().numpy()
                # Take top 4 context channels
                ctx_center = ctx[:, :4, hc, wc].cpu().numpy()

                preds_list.append(p_center)
                err_list.append(err_center)
                ctx_list.append(ctx_center)

        return (
            np.concatenate(preds_list, axis=0),
            np.concatenate(err_list, axis=0),
            np.concatenate(ctx_list, axis=0),
        )

    def save(self, path: Union[Path, str]) -> None:
        """Save fitted DistrictDAEUNet weights to disk."""
        if self.model is None:
            raise RuntimeError("Model is not fitted. Cannot save.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(p))

    def load(self, path: Union[Path, str]) -> "MPMDistrictDAEUNet":
        """Load DistrictDAEUNet weights from disk."""
        self.model = DistrictDAEUNet(
            in_channels=self.in_channels,
            n_classes=self.n_classes,
            dropout_p=self.dropout_p,
            channel_drop_p=self.channel_drop_p,
            px_size_m=self.px_size_m,
            pi_p=self.pi_p,
        ).to(self.device)
        state = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        return self

    def predict(
        self, X_windows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience alias for predict_features."""
        return self.predict_features(X_windows)

    def cross_validate_spatial_blocks(
        self,
        X_grid_50m: np.ndarray,
        y_grid_k4: np.ndarray,
        block_ids_50m: np.ndarray,
        decimate_factor: int = 5,
        epochs: int = 15,
        lr: float = 1e-4,
        checkpoints_dir: Optional[Union[Path, str]] = None,
        checkpoint_prefix: str = "dae_unet",
        propensity_weights_2d: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Train across 5 spatially buffered blocks and project OOF context and error onto native grid.

        Enforces Task 2.1 Rule 3. Supports continuous warm-starting from checkpoints_dir
        and supports both K=4 Metallic and K=3 Coal multi-label heads.

        Args:
            X_grid_50m: (C, H, W) native resolution feature cube.
            y_grid_k4: (K, H, W) binary K-head label grids (K=4 for metals, K=3 for coal).
            block_ids_50m: (H, W) spatial block assignments.
            decimate_factor: Decimation factor (default 5).
            epochs: Training epochs per fold.
            lr: Learning rate.
            checkpoints_dir: Optional directory to load/save fold model weights for incremental learning.
            checkpoint_prefix: Prefix for saved fold checkpoints (e.g. 'dae_metals' or 'dae_coal').
            propensity_weights_2d: Optional (H, W) array of Pass 2 observation propensity weights.

        Returns:
            Dictionary with native-grid projected out-of-fold channels:
              - 'dae_recon_err': (H, W)
              - 'unet_ctx_0' .. 'unet_ctx_3': (H, W)
              - 'oof_prob_k0' .. 'oof_prob_k{K-1}': (H, W)
        """
        C, H_orig, W_orig = X_grid_50m.shape
        target_shape = (H_orig, W_orig)
        n_cls = int(y_grid_k4.shape[0])
        self.n_classes = n_cls
        self.in_channels = int(C)

        # 1. Decimate spatial feature grid
        X_coarse = decimate_spatial_grid(
            X_grid_50m, target_res_m=50.0 * decimate_factor, native_res_m=50.0, order=1
        )
        y_coarse = decimate_spatial_grid(
            y_grid_k4, target_res_m=50.0 * decimate_factor, native_res_m=50.0, order=0
        )
        blocks_coarse = decimate_spatial_grid(
            block_ids_50m.astype(np.int32),
            target_res_m=50.0 * decimate_factor,
            native_res_m=50.0,
            order=0,
        ).astype(np.int32)
        assert set(np.unique(blocks_coarse)).issubset(set(np.unique(block_ids_50m))), (
            "FATAL: Interpolated fold IDs detected in blocks_coarse!"
        )

        C_c, H_c, W_c = X_coarse.shape
        valid_blocks = np.unique(blocks_coarse[blocks_coarse >= 0])
        # Enforce strict 5-fold spatial cross-validation limit as per spec while preserving -1 unassigned cells
        if len(valid_blocks) > 5:
            fold_map = {int(b): int(i % 5) for i, b in enumerate(valid_blocks)}
            fold_map[-1] = -1
            blocks_coarse = np.vectorize(lambda x: fold_map.get(int(x), -1))(blocks_coarse)
            unique_blocks = np.unique(blocks_coarse[blocks_coarse >= 0])
        else:
            unique_blocks = valid_blocks

        oof_recon_err_coarse = np.zeros((H_c, W_c), dtype=np.float32)
        oof_ctx_coarse = np.zeros((4, H_c, W_c), dtype=np.float32)
        oof_preds_coarse = np.zeros((n_cls, H_c, W_c), dtype=np.float32)
        # Decimate 2D observation propensity weights if provided
        if propensity_weights_2d is not None:
            w_coarse = decimate_spatial_grid(
                propensity_weights_2d,
                target_res_m=50.0 * decimate_factor,
                native_res_m=50.0,
                order=1,
            )
            if w_coarse.ndim == 2:
                # Shape (H_c, W_c) -> broadcast across n_cls (H_c * W_c, n_cls)
                w_flat = np.repeat(w_coarse.ravel()[:, np.newaxis], n_cls, axis=1)
            elif w_coarse.ndim == 3:
                w_flat = w_coarse.reshape(w_coarse.shape[0], -1).T
                if w_flat.shape[1] != n_cls:
                    w_flat = np.repeat(w_flat[:, :1], n_cls, axis=1)
            else:
                w_flat = None
        else:
            w_flat = None

        # 2. Setup padding for on-the-fly window extraction (avoids 44GB dense pre-stacking)
        win_size = min(max(16, self.window_px), H_c, W_c)
        pad = win_size // 2
        X_padded = np.pad(X_coarse, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")

        y_flat = y_coarse.reshape(n_cls, -1).T  # (H_c * W_c, n_cls)
        block_flat = blocks_coarse.ravel()

        class _OnTheFlyPatchDataset(torch.utils.data.Dataset):
            def __init__(
                self,
                padded_arr: np.ndarray,
                indices: np.ndarray,
                stride_w: int,
                window_size: int,
                y_arr: Optional[np.ndarray] = None,
                w_arr: Optional[np.ndarray] = None,
            ):
                self.padded = torch.from_numpy(padded_arr).float()
                self.indices = indices
                self.stride_w = stride_w
                self.window_size = window_size
                self.y_arr = torch.from_numpy(y_arr).float() if y_arr is not None else None
                self.w_arr = torch.from_numpy(w_arr).float() if w_arr is not None else None

            def __len__(self) -> int:
                return len(self.indices)

            def __getitem__(self, idx: int):
                c_idx = int(self.indices[idx])
                r = c_idx // self.stride_w
                c = c_idx % self.stride_w
                patch = self.padded[:, r : r + self.window_size, c : c + self.window_size]
                if self.y_arr is not None:
                    y = self.y_arr[idx]
                    if self.w_arr is not None:
                        w = self.w_arr[idx]
                        if w.shape != y.shape:
                            w = w.expand_as(y)
                    else:
                        w = torch.ones_like(y)
                    return patch, y, w
                return patch

        # 3. Train across spatial CV blocks
        for b_id in unique_blocks:
            if b_id < 0:
                continue
            train_mask = (block_flat >= 0) & (block_flat != b_id)
            val_mask = (block_flat == b_id)

            if not np.any(val_mask) or not np.any(train_mask):
                continue

            train_indices = np.where(train_mask)[0]
            # Subsample training coordinates to save memory and balance background cells
            is_pos = np.any(y_flat[train_indices] > 0, axis=1)
            pos_idx = train_indices[is_pos]
            unlabeled_idx = train_indices[~is_pos]

            rng = np.random.default_rng(42 + int(b_id))
            max_pos = 1500
            if len(pos_idx) > max_pos:
                sub_pos = rng.choice(pos_idx, size=max_pos, replace=False)
            else:
                sub_pos = pos_idx

            target_u_count = min(3000, 2 * max(len(sub_pos), 200))
            if len(unlabeled_idx) > target_u_count:
                sub_u = rng.choice(unlabeled_idx, size=target_u_count, replace=False)
            else:
                sub_u = unlabeled_idx
            sub_train_idx = np.concatenate([sub_pos, sub_u])

            fold_model = DistrictDAEUNet(
                in_channels=self.in_channels,
                n_classes=self.n_classes,
                dropout_p=self.dropout_p,
                channel_drop_p=self.channel_drop_p,
                px_size_m=float(self.px_size_m) if self.px_size_m is not None else float(50.0 * decimate_factor),
                pi_p=self.pi_p,
            ).to(self.device)

            # Continuous / Incremental Weight Updates (Warm-Starting):
            fold_ckpt: Optional[Path] = None
            if checkpoints_dir is not None:
                ckpt_dir_path = Path(checkpoints_dir)
                ckpt_dir_path.mkdir(parents=True, exist_ok=True)
                fold_ckpt = ckpt_dir_path / f"{checkpoint_prefix}_fold_{b_id}.pt"
                if fold_ckpt.exists():
                    try:
                        state_dict = torch.load(fold_ckpt, map_location=self.device)
                        fold_model.load_state_dict(state_dict, strict=False)
                        logger.info("Warm-started DAE-U-Net fold %s weights from %s", b_id, fold_ckpt)
                    except Exception as e:
                        logger.warning("Could not warm-start fold %s checkpoint: %s", b_id, e)

            w_sub = (
                w_flat[sub_train_idx, :, np.newaxis, np.newaxis]
                if w_flat is not None
                else None
            )
            train_ds = _OnTheFlyPatchDataset(
                padded_arr=X_padded,
                indices=sub_train_idx,
                stride_w=W_c,
                window_size=win_size,
                y_arr=y_flat[sub_train_idx, :, np.newaxis, np.newaxis],
                w_arr=w_sub,
            )
            eff_train_batch = max(int(self.batch_size), 64)
            loader = DataLoader(
                train_ds,
                batch_size=eff_train_batch,
                shuffle=True,
                drop_last=(len(train_ds) > eff_train_batch),
            )
            optimiser = torch.optim.AdamW(fold_model.parameters(), lr=lr, weight_decay=1e-4)

            fold_model.train()
            for _ in range(epochs):
                for xb, yb, wb in loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    wb = wb.to(self.device)
                    optimiser.zero_grad()
                    p, recon, err, ctx = fold_model(xb)
                    if p.shape[-2:] != (1, 1):
                        hc, wc = p.shape[-2] // 2, p.shape[-1] // 2
                        p = p[:, :, hc : hc + 1, wc : wc + 1]
                    loss_pu = self.criterion(p, yb, sample_weights=wb)
                    if self.subcanopy_indices is not None and len(self.subcanopy_indices) > 0:
                        valid_sub = [i for i in self.subcanopy_indices if i < self.in_channels]
                        if len(valid_sub) > 0:
                            loss_recon = nn.functional.mse_loss(recon[:, valid_sub], xb[:, valid_sub])
                        else:
                            loss_recon = nn.functional.mse_loss(recon, xb)
                    else:
                        loss_recon = nn.functional.mse_loss(recon, xb)
                    loss = loss_pu + self.recon_weight * loss_recon
                    loss.backward()
                    optimiser.step()

            # Save updated fold model weights for future incremental continuity
            if fold_ckpt is not None:
                try:
                    torch.save(fold_model.state_dict(), fold_ckpt)
                    logger.info("Saved updated DAE-U-Net fold %s weights to %s", b_id, fold_ckpt)
                except Exception as e:
                    logger.warning("Could not save fold %s checkpoint: %s", b_id, e)

            # Dense whole-grid inference on coarse feature grid (vectorized OOF projection)
            fold_model.eval()
            pad_h = (16 - (H_c % 16)) % 16
            pad_w = (16 - (W_c % 16)) % 16
            X_c_pad = np.pad(X_coarse, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            with torch.no_grad():
                x_tensor = torch.from_numpy(X_c_pad).unsqueeze(0).float().to(self.device)
                p_full, recon_full, err_full, ctx_full = fold_model(x_tensor)
                p_full = p_full[0, :n_cls, :H_c, :W_c].detach().cpu().numpy()
                err_full = err_full[0, 0, :H_c, :W_c].detach().cpu().numpy()
                ctx_full = ctx_full[0, :4, :H_c, :W_c].detach().cpu().numpy()

            val_mask_2d = (blocks_coarse == b_id)
            oof_recon_err_coarse[val_mask_2d] = err_full[val_mask_2d]
            for k in range(n_cls):
                oof_preds_coarse[k, val_mask_2d] = p_full[k, val_mask_2d]
            for k in range(4):
                oof_ctx_coarse[k, val_mask_2d] = ctx_full[k, val_mask_2d]

            del fold_model, optimiser, train_ds
            import gc
            gc.collect()

        # Zero-positive head handling across entire grid (Bug 3)
        zero_pos_heads = [k for k in range(n_cls) if np.sum(y_grid_k4[k] == 1) == 0]
        for k in zero_pos_heads:
            oof_preds_coarse[k, :] = 0.0

        # 4. Project out-of-fold spatial context embeddings and reconstruction error back to native grid
        proj_err_50m = project_spatial_grid(oof_recon_err_coarse, target_shape=target_shape, order=1)
        proj_ctx_50m = project_spatial_grid(oof_ctx_coarse, target_shape=target_shape, order=1)
        proj_preds_50m = project_spatial_grid(oof_preds_coarse, target_shape=target_shape, order=1)

        for k in zero_pos_heads:
            proj_preds_50m[k] = np.zeros(target_shape, dtype=np.float32)

        result: dict[str, np.ndarray] = {
            "dae_recon_err": proj_err_50m,
        }
        for k in range(4):
            result[f"unet_ctx_{k}"] = proj_ctx_50m[k]
        for k in range(n_cls):
            result[f"oof_prob_k{k}"] = proj_preds_50m[k]

        return result

