"""
src/outputs/model_introspection.py
===================================
Real-data Model Introspection Engine for the SIFT Control Center Dashboard.

Extracts:
  1. Layer-by-layer neuron activations, firing statistics, top active channels,
     and 8x8 2D spatial feature maps from fitted ``DistrictDAEUNet`` checkpoints
     (``dae_unet_fold_*.pt``) across ranked drill targets.
  2. Exact decision tree topologies (split features, thresholds, gain, cover,
     and leaf log-odds weights) from fitted ``TwoStageK4Ensemble`` checkpoints
     (``k4_ensemble.joblib``) across all K=4 deposit heads, alongside each
     target's physical feature vector and active root-to-leaf decision paths.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import pandas as pd
from rasterio.warp import transform as warp_transform
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

K4_HEAD_META = [
    {
        "head_id": 0,
        "head_key": "porphyry_cu",
        "head_label": "HEAD 0 // Porphyry Cu-Au (Stage-2 Meta-Stack)",
        "short_label": "H0: Porphyry Cu-Au",
        "max_depth": 3,
    },
    {
        "head_id": 1,
        "head_key": "skarn_fe_cu",
        "head_label": "HEAD 1 // Skarn Fe-Cu (Contact Metasomatic)",
        "short_label": "H1: Skarn Fe-Cu",
        "max_depth": 6,
    },
    {
        "head_id": 2,
        "head_key": "epithermal_au",
        "head_label": "HEAD 2 // Epithermal Au-Ag (Structural Vein)",
        "short_label": "H2: Epithermal Au-Ag",
        "max_depth": 6,
    },
    {
        "head_id": 3,
        "head_key": "laterite_fe_ni",
        "head_label": "HEAD 3 // Lateritic Fe-Ni (Ultramafic Regolith)",
        "short_label": "H3: Lateritic Fe-Ni",
        "max_depth": 6,
    },
]

UNET_LAYER_CATALOG = [
    {"id": "input", "label": "Input Patch", "group": "INPUT", "channels": 21, "res": "32×32", "params": 0, "mb": 0.00},
    {"id": "inc", "label": "Enc1 DoubleConv", "group": "ENCODER", "channels": 64, "res": "32×32", "params": 49280, "mb": 0.20},
    {"id": "down1", "label": "Enc2 Down1", "group": "ENCODER", "channels": 128, "res": "16×16", "params": 221696, "mb": 0.89},
    {"id": "down2", "label": "Enc3 Down2", "group": "ENCODER", "channels": 256, "res": "8×8", "params": 885760, "mb": 3.54},
    {"id": "down3", "label": "Enc4 Down3", "group": "ENCODER", "channels": 512, "res": "4×4", "params": 3540992, "mb": 14.16},
    {"id": "down4", "label": "Bottleneck + Drop2d", "group": "BOTTLENECK", "channels": 1024, "res": "2×2", "params": 14159872, "mb": 56.64},
    {"id": "up1", "label": "Dec1 Up1 + Skip4", "group": "SUP_DECODER", "channels": 512, "res": "4×4", "params": 14158336, "mb": 56.63},
    {"id": "up2", "label": "Dec2 Up2 + Skip3", "group": "SUP_DECODER", "channels": 256, "res": "8×8", "params": 3540224, "mb": 14.16},
    {"id": "up3", "label": "Dec3 Up3 + Skip2", "group": "SUP_DECODER", "channels": 128, "res": "16×16", "params": 885376, "mb": 3.54},
    {"id": "up4", "label": "Dec4 Up4 + Skip1", "group": "SUP_DECODER", "channels": 64, "res": "32×32", "params": 221504, "mb": 0.89},
    {"id": "sa", "label": "Spatial Attention (7×7)", "group": "ATTENTION", "channels": 64, "res": "32×32", "params": 98, "mb": 0.00},
    {"id": "out_heads", "label": "K=4 Heads Sigmoid", "group": "HEADS", "channels": 4, "res": "32×32", "params": 260, "mb": 0.00},
    {"id": "dae_up1", "label": "DAE Up1 (No Skip)", "group": "DAE_DECODER", "channels": 512, "res": "4×4", "params": 7079936, "mb": 28.32},
    {"id": "dae_up2", "label": "DAE Up2 (No Skip)", "group": "DAE_DECODER", "channels": 256, "res": "8×8", "params": 1770496, "mb": 7.08},
    {"id": "dae_up3", "label": "DAE Up3 (No Skip)", "group": "DAE_DECODER", "channels": 128, "res": "16×16", "params": 442880, "mb": 1.77},
    {"id": "dae_up4", "label": "DAE Up4 (No Skip)", "group": "DAE_DECODER", "channels": 64, "res": "32×32", "params": 110848, "mb": 0.44},
    {"id": "dae_recon", "label": "DAE Recon (21ch)", "group": "DAE_DECODER", "channels": 21, "res": "32×32", "params": 1365, "mb": 0.01},
]

from src.schema.canonical import (
    NON_FEATURE_COLS as CANONICAL_NON_FEATURE_COLS,
    EXTENDED_NON_FEATURE_COLS,
    NON_FEATURE_COLS_SET,
    _FORBIDDEN_RAW_BAND_PATTERNS,
    FORBIDDEN_RAW_BAND_REGEXES,
    K4_HEAD_NAMES,
    verify_no_raw_pixel_leakage,
)

# Centralized non-feature columns set
NON_FEATURE_COLS = set(EXTENDED_NON_FEATURE_COLS) | set(CANONICAL_NON_FEATURE_COLS) | {
    "alteration_benchmark",
    "commodity_label",
    "is_centroid_artifact",
    "target_class",
    "w_canopy",
}


def load_dae_unet_model(
    checkpoint_path: Path | str,
    in_channels: int = 21,
    n_classes: int = 4,
    device: str = "cpu",
) -> Any:
    """Load a trained DistrictDAEUNet PyTorch model from a checkpoint file."""
    import torch
    from src.models.unet_model import DistrictDAEUNet

    model = DistrictDAEUNet(in_channels=in_channels, n_classes=n_classes)
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state_fp32 = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in state.items()}
    model.load_state_dict(state_fp32, strict=False)
    model.eval()
    return model



def _enrich_xgb_tree_node(node: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    """Recursively enrich an XGBoost JSON tree node with human-readable feature names."""
    out: dict[str, Any] = {
        "nodeid": int(node.get("nodeid", 0)),
        "depth": int(node.get("depth", 0)),
    }
    if "leaf" in node:
        out["is_leaf"] = True
        out["leaf"] = round(float(node["leaf"]), 5)
        out["cover"] = round(float(node.get("cover", 0.0)), 1)
        return out

    out["is_leaf"] = False
    raw_split = str(node.get("split", "f0"))
    feat_idx = 0
    if raw_split.startswith("f") and raw_split[1:].isdigit():
        feat_idx = int(raw_split[1:])
    feat_name = feature_names[feat_idx] if 0 <= feat_idx < len(feature_names) else raw_split

    out["feature_code"] = raw_split
    out["feature_idx"] = feat_idx
    out["feature_name"] = feat_name
    out["split_condition"] = round(float(node.get("split_condition", 0.0)), 4)
    out["yes"] = int(node.get("yes", 0))
    out["no"] = int(node.get("no", 0))
    out["missing"] = int(node.get("missing", out["no"]))
    out["gain"] = round(float(node.get("gain", 0.0)), 1)
    out["cover"] = round(float(node.get("cover", 0.0)), 1)
    children = node.get("children", [])
    out["children"] = [_enrich_xgb_tree_node(ch, feature_names) for ch in children]
    return out


def trace_xgb_decision_path(tree_root: dict[str, Any], feature_vec: list[float]) -> dict[str, Any]:
    """Trace a feature vector from root to leaf in an enriched XGBoost tree."""
    active_nodes: list[int] = []
    steps: list[dict[str, Any]] = []
    curr = tree_root

    while curr is not None:
        nid = int(curr["nodeid"])
        active_nodes.append(nid)
        if curr.get("is_leaf"):
            return {
                "active_nodes": active_nodes,
                "leaf_nodeid": nid,
                "leaf_weight": float(curr.get("leaf", 0.0)),
                "steps": steps,
            }
        f_idx = int(curr.get("feature_idx", 0))
        f_name = str(curr.get("feature_name", f"f{f_idx}"))
        cond = float(curr.get("split_condition", 0.0))
        val = float(feature_vec[f_idx]) if 0 <= f_idx < len(feature_vec) else float("nan")
        if np.isnan(val):
            next_id = int(curr.get("missing", curr.get("no", 0)))
            direction = "MISSING"
        elif val < cond:
            next_id = int(curr.get("yes", 0))
            direction = "YES (<)"
        else:
            next_id = int(curr.get("no", 0))
            direction = "NO (>=)"

        steps.append({
            "nodeid": nid,
            "feature_name": f_name,
            "feature_val": round(val, 4) if np.isfinite(val) else None,
            "threshold": cond,
            "direction": direction,
            "next_nodeid": next_id,
            "gain": float(curr.get("gain", 0.0)),
        })
        children = curr.get("children", [])
        curr = next((c for c in children if int(c["nodeid"]) == next_id), None)

    return {"active_nodes": active_nodes, "leaf_nodeid": active_nodes[-1] if active_nodes else 0, "leaf_weight": 0.0, "steps": steps}


def _downsample_to_8x8(arr_2d: np.ndarray) -> list[list[float]]:
    """Pool/interpolate a 2D activation map to an 8x8 normalized [0, 1] grid for UI rendering."""
    h, w = arr_2d.shape
    if h == 8 and w == 8:
        grid = arr_2d.copy()
    else:
        grid = zoom(arr_2d.astype(np.float32), (8.0 / max(h, 1), 8.0 / max(w, 1)), order=1)[:8, :8]
    vmin = float(np.min(grid))
    vmax = float(np.max(grid))
    if vmax - vmin > 1e-6:
        norm = (grid - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(grid)
    return [[round(float(v), 3) for v in row] for row in norm]


def _eval_tree_leaf(node: dict[str, Any], feat_vec: np.ndarray) -> float:
    """Evaluate a single raw XGBoost JSON tree dict on a 1D feature vector."""
    curr = node
    while curr is not None:
        if "leaf" in curr:
            return float(curr["leaf"])
        raw_split = str(curr.get("split", "f0"))
        f_idx = int(raw_split[1:]) if (raw_split.startswith("f") and raw_split[1:].isdigit()) else 0
        cond = float(curr.get("split_condition", 0.0))
        val = float(feat_vec[f_idx]) if 0 <= f_idx < len(feat_vec) else float("nan")
        if np.isnan(val):
            next_id = int(curr.get("missing", curr.get("no", 0)))
        elif val < cond:
            next_id = int(curr.get("yes", 0))
        else:
            next_id = int(curr.get("no", 0))
        children = curr.get("children", [])
        curr = next((c for c in children if int(c["nodeid"]) == next_id), None)
    return 0.0


def _run_unet_hooks_isolated(
    unet_ckpt: Path,
    patches: np.ndarray,
    target_ids: list[str],
    feature_cols: list[str],
) -> dict[str, dict[str, Any]]:
    """Run DistrictDAEUNet forward hooks in an isolated Python subprocess to avoid libomp collisions with XGBoost."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        patches_path = tmp_p / "patches.npy"
        out_path = tmp_p / "unet_acts.json"
        np.save(patches_path, patches)

        code = f"""
import json, numpy as np, torch
from pathlib import Path
from scipy.ndimage import zoom
from src.models.unet_model import DistrictDAEUNet

torch.set_num_threads(4)
patches = np.load({str(patches_path)!r})
target_ids = {target_ids!r}
feature_cols = {feature_cols!r}
C = patches.shape[1]

def downsample_8x8(arr_2d):
    h, w = arr_2d.shape
    grid = arr_2d.copy() if (h == 8 and w == 8) else zoom(arr_2d.astype(np.float32), (8.0 / max(h, 1), 8.0 / max(w, 1)), order=1)[:8, :8]
    vmin, vmax = float(np.min(grid)), float(np.max(grid))
    norm = (grid - vmin) / (vmax - vmin) if (vmax - vmin > 1e-6) else np.zeros_like(grid)
    return [[round(float(v), 3) for v in row] for row in norm]

model = DistrictDAEUNet(in_channels=C, n_classes=4)
state = torch.load({str(unet_ckpt)!r}, map_location="cpu", weights_only=True)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
state_fp32 = {{k: (v.float() if torch.is_floating_point(v) else v) for k, v in state.items()}}
model.load_state_dict(state_fp32)
model.eval()

hook_outputs = {{}}
hook_layer_names = ["inc", "down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4", "sa", "out_heads", "dae_up1", "dae_up2", "dae_up3", "dae_up4", "dae_recon"]
for lname in hook_layer_names:
    getattr(model, lname).register_forward_hook(lambda m, inp, out, n=lname: hook_outputs.update({{n: out.detach().cpu().numpy()}}))

with torch.no_grad():
    preds_t, recon_t, recon_err_t, ctx_t = model(torch.from_numpy(patches))
    preds_np = preds_t.cpu().numpy()
    recon_err_np = recon_err_t.cpu().numpy()

res = {{}}
for b_i, tid in enumerate(target_ids):
    layers_dict = {{}}
    in_patch = patches[b_i]
    ch_means_in = np.mean(np.abs(in_patch), axis=(1, 2))
    top_ch_in = np.argsort(ch_means_in)[::-1][:6]
    c_max_in = float(np.max(ch_means_in)) + 1e-6
    layers_dict["input"] = {{
        "mean_act": round(float(np.mean(np.abs(in_patch))), 3),
        "max_act": round(float(np.max(np.abs(in_patch))), 3),
        "active_pct": round(float(np.mean(np.abs(in_patch) > 0.25) * 100.0), 1),
        "top_neurons": [
            {{
                "ch": int(ch_i),
                "name": feature_cols[int(ch_i)] if int(ch_i) < len(feature_cols) else f"ch_{{ch_i}}",
                "act": round(float(ch_means_in[ch_i]), 3),
                "norm": round(float(ch_means_in[ch_i] / c_max_in), 3),
            }}
            for ch_i in top_ch_in
        ],
        "grid_8x8": downsample_8x8(in_patch[int(top_ch_in[0])]),
    }}
    head_names_map = {{0: "porphyry_cu", 1: "skarn_fe_cu", 2: "epithermal_au", 3: "laterite_fe_ni"}}
    for lname in hook_layer_names:
        arr = hook_outputs[lname][b_i]
        abs_arr = np.abs(arr) if lname in ("out_heads", "dae_recon") else np.maximum(arr, 0.0)
        ch_means = np.mean(abs_arr, axis=(1, 2))
        top_idx = np.argsort(ch_means)[::-1][:6]
        c_max = float(np.max(ch_means)) + 1e-6
        active_frac = float(np.mean(arr > 0.05) * 100.0)
        top_neurons = []
        for ch_i in top_idx:
            ch_int = int(ch_i)
            if lname == "out_heads":
                n_label = head_names_map.get(ch_int, f"head_{{ch_int}}")
            elif lname == "dae_recon" and ch_int < len(feature_cols):
                n_label = feature_cols[ch_int]
            else:
                n_label = f"neuron_{{ch_int:03d}}"
            top_neurons.append({{
                "ch": ch_int,
                "name": n_label,
                "act": round(float(ch_means[ch_int]), 3),
                "norm": round(float(ch_means[ch_int] / c_max), 3),
            }})
        dom_ch = int(top_idx[0])
        layers_dict[lname] = {{
            "mean_act": round(float(np.mean(abs_arr)), 3),
            "max_act": round(float(np.max(abs_arr)), 3),
            "active_pct": round(active_frac, 1),
            "top_neurons": top_neurons,
            "grid_8x8": downsample_8x8(abs_arr[dom_ch]),
        }}
    hc, wc = preds_np.shape[-2] // 2, preds_np.shape[-1] // 2
    res[tid] = {{
        "unet_probs_k4": [round(float(v), 4) for v in preds_np[b_i, :, hc, wc]],
        "dae_recon_mse": round(float(recon_err_np[b_i, 0, hc, wc]), 4),
        "layers": layers_dict,
    }}

Path({str(out_path)!r}).write_text(json.dumps(res), encoding="utf-8")
"""
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        subprocess.run([sys.executable, "-c", code], check=True, env=env)
        return json.loads(out_path.read_text(encoding="utf-8"))


def build_model_introspection_payload(
    data_dir: Path,
    out_dir: Path,
    region_name: str = "EAST_KALIMANTAN",
    max_trees_per_head: int = 8,
) -> dict[str, Any]:
    """Generate authentic U-Net neuron activation & XGBoost decision tree payload from disk checkpoints."""
    models_dir = out_dir / "models"
    cards_path = out_dir / "targets" / "site_cards.json"
    parquet_path = data_dir / "dataset.parquet"
    stack_path = data_dir / "feature_stack.joblib"

    df = pd.read_parquet(parquet_path)
    stack = joblib.load(stack_path)
    profile = stack.profile

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and not c.startswith("label_")
        and not any(pat in c.lower() for pat in _FORBIDDEN_RAW_BAND_PATTERNS)
    ]
    verify_no_raw_pixel_leakage(feature_cols)
    # Cap distance features at local physical limit (15.0 km / 15000.0 m)
    for c in feature_cols:
        if c.startswith("dist_") or c.endswith("_dist_km"):
            if c.endswith("_m"):
                df[c] = np.clip(df[c].to_numpy(dtype=np.float32), 0.0, 15000.0)
            else:
                df[c] = np.clip(df[c].to_numpy(dtype=np.float32), 0.0, 15.0)

    head0_feature_cols = list(feature_cols) + ["stage1_prob_skarn_fe_cu", "stage1_prob_epithermal_au"]

    # Filter stack channels to strictly match feature_cols (excluding NON_FEATURE_COLS and leaked halos)
    stack_feats = list(stack.feature_names) if hasattr(stack, "feature_names") else feature_cols
    feat_indices = [i for i, name in enumerate(stack_feats) if name in feature_cols]
    if feat_indices:
        cube_raw = stack.array[feat_indices].astype(np.float32)
    else:
        valid_indices = [i for i, f in enumerate(stack_feats) if f not in NON_FEATURE_COLS and not f.startswith("label_")]
        cube_raw = stack.array[valid_indices].astype(np.float32)

    C, H, W = cube_raw.shape
    for i, name in enumerate(feature_cols):
        if i < C and (name.startswith("dist_") or name.endswith("_dist_km")):
            if name.endswith("_m"):
                cube_raw[i] = np.clip(cube_raw[i], 0.0, 15000.0)
            else:
                cube_raw[i] = np.clip(cube_raw[i], 0.0, 15.0)

    site_cards = json.loads(cards_path.read_text(encoding="utf-8")) if cards_path.exists() else []
    rows_arr = df["row"].to_numpy(dtype=np.int32)
    cols_arr = df["col"].to_numpy(dtype=np.int32)
    X_all = df[feature_cols].to_numpy(dtype=np.float32)

    # Fast channel normalization statistics from strided feature matrix (2ms vs 20s full-cube scan)
    X_sub = X_all[::20]
    ch_means_norm = np.nanmean(X_sub, axis=0).astype(np.float32)
    ch_stds_norm = (np.nanstd(X_sub, axis=0) + 1e-6).astype(np.float32)

    target_meta: list[dict[str, Any]] = []
    inv_tr = ~profile["transform"]
    src_crs = str(profile["crs"])

    for idx, sc in enumerate(site_cards[:15]):
        tid = str(sc.get("target_id", f"TGT_{idx+1:04d}"))
        lat, lon = sc.get("geometry", {}).get("centroid_lat_lon", [-0.5, 116.0])
        if src_crs.upper() != "EPSG:4326":
            xs, ys = warp_transform("EPSG:4326", src_crs, [lon], [lat])
            px_x, px_y = xs[0], ys[0]
        else:
            px_x, px_y = lon, lat
        col_f, row_f = inv_tr * (px_x, px_y)
        r_idx = int(np.clip(round(row_f), 0, H - 1))
        c_idx = int(np.clip(round(col_f), 0, W - 1))

        match_pix = np.where((rows_arr == r_idx) & (cols_arr == c_idx))[0]
        if len(match_pix) > 0:
            df_idx = int(match_pix[0])
        else:
            df_idx = int(np.argmin((rows_arr - r_idx) ** 2 + (cols_arr - c_idx) ** 2))
            r_idx = int(rows_arr[df_idx])
            c_idx = int(cols_arr[df_idx])

        target_meta.append({
            "target_id": tid,
            "rank": int(sc.get("rank", idx + 1)),
            "lat": round(float(lat), 4),
            "lon": round(float(lon), 4),
            "row": r_idx,
            "col": c_idx,
            "df_idx": df_idx,
            "calibrated_prob": round(float(sc.get("probabilities", {}).get("calibrated_deposit_prob", sc.get("rank_score", 0.9))), 4),
            "ore_category": str(sc.get("ore_category") or (sc.get("conformal_90pct_target_set") or ["epithermal_au"])[0]),
        })

    # 1. Extract Real PyTorch DistrictDAEUNet Layer Activations in isolated subprocess
    unet_ckpt = models_dir / "dae_unet_fold_0.pt"
    cands = (
        sorted(models_dir.glob("dae_metals_fold_*.pt"))
        + sorted(models_dir.glob("dae_coal_fold_*.pt"))
        + sorted(models_dir.glob("dae_unet_fold_*.pt"))
    )
    if cands:
        unet_ckpt = cands[0]

    target_unet_acts: dict[str, dict[str, Any]] = {}
    if unet_ckpt.exists() and target_meta:
        win_px = 32
        half = win_px // 2
        patches_list = []
        for t in target_meta:
            r0 = int(np.clip(t["row"] - half, 0, max(0, H - win_px)))
            c0 = int(np.clip(t["col"] - half, 0, max(0, W - win_px)))
            raw_win = np.asarray(cube_raw[:, r0 : r0 + win_px, c0 : c0 + win_px], dtype=np.float32)
            if raw_win.shape[1] < win_px or raw_win.shape[2] < win_px:
                raw_win = np.pad(
                    raw_win,
                    ((0, 0), (0, max(0, win_px - raw_win.shape[1])), (0, max(0, win_px - raw_win.shape[2]))),
                    mode="edge",
                )
            norm_win = np.nan_to_num(
                (raw_win - ch_means_norm[:, None, None]) / ch_stds_norm[:, None, None],
                nan=0.0,
            )
            patches_list.append(norm_win)
        patches = np.stack(patches_list, axis=0)
        target_unet_acts = _run_unet_hooks_isolated(
            unet_ckpt=unet_ckpt,
            patches=patches,
            target_ids=[t["target_id"] for t in target_meta],
            feature_cols=feature_cols,
        )

    # 2. Extract XGBoost Trees & Compute Per-Target Feature Vectors
    ens_path = models_dir / "k4_ensemble.joblib"
    xgb_heads_out: list[dict[str, Any]] = []
    target_xgb_vectors: dict[str, list[float]] = {}
    target_xgb_probs: dict[str, list[float]] = {}

    if ens_path.exists():
        ens = joblib.load(ens_path)
        parsed_head_trees: dict[int, list[dict[str, Any]]] = {}
        for meta in K4_HEAD_META:
            k = meta["head_id"]
            h_obj = getattr(ens, f"head_{k}_", None)
            f_names = head0_feature_cols if k == 0 else feature_cols
            if h_obj is not None and getattr(h_obj, "models_", None) and not getattr(h_obj, "is_constant_zero_", False):
                m0 = h_obj.models_[0]
                booster = m0.get_booster()
                dumps = booster.get_dump(with_stats=True, dump_format="json")
                n_total = len(dumps)
                trees_list = []
                raw_trees_k = []
                chosen_indices = list(range(min(max_trees_per_head, n_total)))
                for t_i in chosen_indices:
                    raw_tree = json.loads(dumps[t_i])
                    raw_trees_k.append(raw_tree)
                    enriched_root = _enrich_xgb_tree_node(raw_tree, f_names)
                    trees_list.append({
                        "tree_idx": t_i,
                        "root": enriched_root,
                    })
                parsed_head_trees[k] = raw_trees_k
                xgb_heads_out.append({
                    **meta,
                    "is_active": True,
                    "status_note": f"{n_total} boosted trees × {len(h_obj.models_)} bags fitted on {region_name}",
                    "n_total_trees": n_total,
                    "n_bags": len(h_obj.models_),
                    "learning_rate": round(float(getattr(h_obj, "effective_learning_rate_", 0.05)), 4),
                    "feature_names": f_names,
                    "trees": trees_list,
                })
            else:
                parsed_head_trees[k] = []
                xgb_heads_out.append({
                    **meta,
                    "is_active": False,
                    "status_note": f"Constant Zero Prior (0 positive {meta['short_label']} training occurrences in {region_name} split)",
                    "n_total_trees": 0,
                    "n_bags": 0,
                    "learning_rate": 0.0,
                    "feature_names": f_names,
                    "trees": [],
                })

        for t in target_meta:
            x_row = X_all[t["df_idx"]]
            logits_s1 = {}
            for k_s1 in (1, 2, 3):
                trees_k = parsed_head_trees.get(k_s1, [])
                if trees_k:
                    s_val = sum(_eval_tree_leaf(tr, x_row) for tr in trees_k)
                    logits_s1[k_s1] = float(1.0 / (1.0 + np.exp(-np.clip(s_val, -15.0, 15.0))))
                else:
                    logits_s1[k_s1] = 0.0
            x_full = np.concatenate([x_row, np.array([logits_s1[1], logits_s1[2]], dtype=np.float32)])
            trees_k0 = parsed_head_trees.get(0, [])
            if trees_k0:
                s0 = sum(_eval_tree_leaf(tr, x_full) for tr in trees_k0)
                p0 = float(1.0 / (1.0 + np.exp(-np.clip(s0, -15.0, 15.0))))
            else:
                p0 = 0.0
            target_xgb_vectors[t["target_id"]] = [round(float(v), 4) for v in x_full]
            target_xgb_probs[t["target_id"]] = [
                round(p0, 4),
                round(logits_s1[1], 4),
                round(logits_s1[2], 4),
                round(logits_s1[3], 4),
            ]
        del ens

    # Combine per-target records
    targets_payload: dict[str, Any] = {}
    for t in target_meta:
        tid = t["target_id"]
        f_vec = target_xgb_vectors.get(tid, [0.0] * len(head0_feature_cols))
        paths_preview: dict[str, Any] = {}
        for h_info in xgb_heads_out:
            if h_info["is_active"] and h_info["trees"]:
                paths_preview[str(h_info["head_id"])] = trace_xgb_decision_path(
                    h_info["trees"][0]["root"], f_vec
                )
        targets_payload[tid] = {
            **t,
            "feature_vector": f_vec,
            "feature_dict": {
                head0_feature_cols[i]: f_vec[i]
                for i in range(min(len(head0_feature_cols), len(f_vec)))
            },
            "xgb_probs_k4": target_xgb_probs.get(tid, [0.0, 0.0, 0.0, 0.0]),
            "xgb_tree0_paths": paths_preview,
            "unet": target_unet_acts.get(tid, {}),
        }

    return {
        "status": "ok",
        "region": region_name,
        "checkpoint_unet": str(unet_ckpt.name) if unet_ckpt.exists() else None,
        "checkpoint_xgb": str(ens_path.name) if ens_path.exists() else None,
        "total_unet_params": sum(item["params"] for item in UNET_LAYER_CATALOG),
        "total_unet_mb": round(sum(item["mb"] for item in UNET_LAYER_CATALOG), 2),
        "feature_names": feature_cols,
        "head0_feature_names": head0_feature_cols,
        "unet_layer_catalog": UNET_LAYER_CATALOG,
        "xgb_heads": xgb_heads_out,
        "targets": targets_payload,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    for reg, d_sub, o_sub in [
        ("EAST_KALIMANTAN", "data/east_kalimantan/processed", "outputs/seven_pass_east_kalimantan"),
        ("BORNEO", "data/borneo/processed", "outputs/seven_pass_borneo"),
    ]:
        d_path = root / d_sub
        o_path = root / o_sub
        if (d_path / "dataset.parquet").exists() and (o_path / "models").exists():
            payload = build_model_introspection_payload(d_path, o_path, region_name=reg)
            out_json = o_path / "models" / "model_introspection.json"
            out_json.write_text(json.dumps(payload), encoding="utf-8")
            print(f"[{reg}] Wrote {out_json} ({out_json.stat().st_size / 1024:.1f} KB, {len(payload['targets'])} targets)")
