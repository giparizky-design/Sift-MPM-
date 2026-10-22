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
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


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
        pi_p: float = 0.05,
        gamma: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        super().__init__()
        if not (0.0 < pi_p < 1.0):
            raise ValueError(f"pi_p must be in (0, 1), got {pi_p}")
        self.pi_p: float = float(pi_p)
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
        """Compute nnPU risk loss on sigmoid probabilities ``preds``.

        Args:
            preds: Predicted probabilities in $[0, 1]$, shape ``(B, ...)``.
            targets: Binary/PU labels where ``1`` = positive ($P$) and ``0`` or
                ``-1`` = unlabeled ($U$), broadcastable to ``preds``.
            sample_weights: Optional inverse propensity weights $w_i = 1 / e(x_i)$,
                broadcastable to ``preds``.

        Returns:
            Scalar nnPU loss tensor.
        """
        eps = 1e-7
        p = torch.clamp(preds, eps, 1.0 - eps)
        t = targets.expand_as(p)

        if sample_weights is not None:
            w = sample_weights.expand_as(p)
        else:
            w = torch.ones_like(p)

        pos_mask = (t > 0.5).float()
        unl_mask = 1.0 - pos_mask

        # Surrogate loss: binary cross-entropy l(p, +1) = -log(p), l(p, -1) = -log(1 - p)
        loss_pos = -torch.log(p)
        loss_neg = -torch.log(1.0 - p)

        pos_denom = torch.clamp(torch.sum(pos_mask * w), min=eps)
        unl_denom = torch.clamp(torch.sum(unl_mask * w), min=eps)

        has_pos = torch.sum(pos_mask) > 0
        has_unl = torch.sum(unl_mask) > 0

        r_p_plus = (
            torch.sum(pos_mask * w * loss_pos) / pos_denom
            if has_pos
            else torch.zeros((), device=p.device, dtype=p.dtype)
        )
        r_p_minus = (
            torch.sum(pos_mask * w * loss_neg) / pos_denom
            if has_pos
            else torch.zeros((), device=p.device, dtype=p.dtype)
        )
        r_u_minus = (
            torch.sum(unl_mask * w * loss_neg) / unl_denom
            if has_unl
            else torch.zeros((), device=p.device, dtype=p.dtype)
        )

        r_neg = r_u_minus - self.pi_p * r_p_minus
        self.last_unlabeled_risk = float(r_neg.detach().cpu().item())

        if r_neg.item() < -self.beta:
            self.last_clamped = True
            clamped_forward = self.pi_p * r_p_plus
            if self.gamma > 0.0:
                # Forward value is strictly clamped at pi_p * R_p^+ (>= 0),
                # while backward gradient is -gamma * d(r_neg)/d(theta) (gradient reversal)
                reversed_grad_term = -self.gamma * r_neg
                return clamped_forward.detach() + (
                    reversed_grad_term - reversed_grad_term.detach()
                )
            return clamped_forward
        else:
            self.last_clamped = False
            return self.pi_p * r_p_plus + r_neg



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

        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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
        active_pi_p = self.pi_p if pi_p is None else float(pi_p)

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
