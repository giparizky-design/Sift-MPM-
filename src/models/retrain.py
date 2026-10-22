"""Controlled retraining and versioning pipeline for Mineral Prospectivity Mapping.

The model does NOT learn continuously. Unverified AI predictions MUST NEVER
become training labels (catastrophic feedback loop). Only assay-verified
drillhole results reviewed by a licensed geologist enter the label lake.

Geoscientific rationale:
    Mineral prospectivity models are trained on sparse, high-value ground-truth
    data (drillhole assays). If model predictions are recycled back as training
    labels the model will amplify its own biases, causing it to systematically
    overpredict in areas it already favours and underpredict elsewhere. This
    renders the model useless as an exploration decision-support tool and can
    result in misallocation of capital. Every label that enters the label lake
    must trace back to a physical sample with a chain of custody: collar survey →
    core logging → laboratory assay → geologist sign-off.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class FeedbackLoopError(RuntimeError):
    """Raised when AI-derived predictions are detected in the label lake.

    This is a hard safety guard. AI predictions must NEVER become training
    labels. If this exception is raised the label lake is contaminated and
    must be audited by a licensed geologist before retraining can proceed.
    """


class MissingReviewerError(ValueError):
    """Raised when new drillhole data is submitted without a human reviewer ID.

    Every new label entering the label lake requires a signed-off reviewer
    (licensed geologist) to maintain chain of custody.
    """


class RetrainingPipeline:
    """Orchestrates controlled, versioned retraining of the MPM model.

    This class enforces strict data governance rules:

    1. **Label lake immutability** — parquet files in ``label_lake_dir`` are
       read-only after being written; new labels are *appended* in a new
       timestamped shard, never modifying existing shards.
    2. **Feedback-loop guard** — any row with ``label_source == "ai_prediction"``
       immediately raises :class:`FeedbackLoopError`.
    3. **Human sign-off** — ``reviewer_id`` is mandatory for every batch of new
       labels; it is stored alongside each row for full auditability.
    4. **From-scratch retraining** — no warm-starting from a prior model
       checkpoint; the model is always retrained from the full label lake to
       prevent silent catastrophic forgetting or distribution shift.

    Attributes:
        config (dict): Pipeline configuration dictionary (loaded from pilot.yaml).
        label_lake_dir (Path): Directory containing immutable label lake parquet shards.
        model_dir (Path): Root directory for versioned model artefacts.
    """

    # Sentinel value that must NEVER appear as a label_source in the lake.
    _FORBIDDEN_SOURCE: str = "ai_prediction"

    def __init__(
        self,
        config: dict,
        label_lake_dir: str,
        model_dir: str,
    ) -> None:
        """Initialise the retraining pipeline.

        Args:
            config: Flat or nested configuration dictionary. Must contain at
                minimum ``model.type`` (e.g. ``"xgboost"`` or ``"rf"``),
                ``cv.n_folds``, and ``cv.spatial_block_col`` keys.
            label_lake_dir: Absolute or relative path to the label lake
                directory. All ``*.parquet`` files in this directory are
                treated as label lake shards. The directory must exist.
            model_dir: Absolute or relative path to the root model artefact
                directory. Versioned subdirectories will be created here.

        Raises:
            FileNotFoundError: If ``label_lake_dir`` does not exist.
        """
        self.config = config
        self.label_lake_dir = Path(label_lake_dir)
        self.model_dir = Path(model_dir)

        if not self.label_lake_dir.exists():
            raise FileNotFoundError(
                f"Label lake directory does not exist: {self.label_lake_dir}"
            )

        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "RetrainingPipeline initialised | label_lake=%s | model_dir=%s",
            self.label_lake_dir,
            self.model_dir,
        )

    # ------------------------------------------------------------------
    # Label lake I/O
    # ------------------------------------------------------------------

    def load_label_lake(self) -> pd.DataFrame:
        """Read and validate the full label lake from all parquet shards.

        Loads every ``*.parquet`` file in ``self.label_lake_dir``, concatenates
        them, and performs the critical feedback-loop guard: if *any* row
        carries ``label_source == "ai_prediction"`` the method raises
        immediately without returning data.

        Returns:
            DataFrame with columns at minimum:
            ``["easting", "northing", "label", "label_source",
               "reviewer_id", "timestamp", "assay_id"]``.

        Raises:
            FileNotFoundError: If no parquet shards are found.
            FeedbackLoopError: If any row has ``label_source == "ai_prediction"``.
                This indicates label lake contamination and must be resolved
                by a licensed geologist before retraining.
        """
        shards = sorted(self.label_lake_dir.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"No parquet shards found in label lake: {self.label_lake_dir}"
            )

        logger.info("Loading %d label lake shard(s)…", len(shards))
        frames = [pd.read_parquet(p) for p in shards]
        lake = pd.concat(frames, ignore_index=True)

        # ── CRITICAL SAFETY GUARD ──────────────────────────────────────
        # AI predictions must NEVER be used as training labels.
        contaminated = lake[lake["label_source"] == self._FORBIDDEN_SOURCE]
        if not contaminated.empty:
            n = len(contaminated)
            sample_ids = contaminated["assay_id"].head(5).tolist()
            raise FeedbackLoopError(
                f"LABEL LAKE CONTAMINATION DETECTED: {n} row(s) have "
                f"label_source='{self._FORBIDDEN_SOURCE}'. This indicates AI "
                f"predictions have been injected into the training data — a "
                f"catastrophic feedback loop. The label lake must be audited "
                f"by a licensed geologist. Offending assay_ids (first 5): "
                f"{sample_ids}"
            )
        # ───────────────────────────────────────────────────────────────

        logger.info(
            "Label lake loaded: %d rows | sources: %s",
            len(lake),
            lake["label_source"].value_counts().to_dict(),
        )
        return lake

    def prepare_new_labels(
        self,
        new_drillhole_csv: str,
        reviewer_id: str,
    ) -> None:
        """Validate and append new drillhole assay results to the label lake.

        Reads a CSV of new drillhole assay results, enforces human sign-off via
        ``reviewer_id``, stamps every row with a UTC timestamp and
        ``label_source="field_verified"``, then writes a new timestamped parquet
        shard. Existing shards are never modified (append-only lake).

        The expected CSV schema is::

            assay_id, easting, northing, depth_from, depth_to,
            au_ppm, cu_pct, label (P/RN/U)

        Args:
            new_drillhole_csv: Path to the CSV file containing new assay
                results. Must include ``assay_id`` and ``label`` columns.
            reviewer_id: Unique identifier of the licensed geologist who
                reviewed and signed off these assay results. Cannot be empty.
                This is stored verbatim for audit trail purposes.

        Raises:
            MissingReviewerError: If ``reviewer_id`` is empty or whitespace-only.
            FileNotFoundError: If ``new_drillhole_csv`` does not exist.
            ValueError: If required columns are missing from the CSV.

        Returns:
            None. A new parquet shard is written to ``self.label_lake_dir``.
        """
        # Enforce human sign-off — no automated batch submission allowed.
        if not reviewer_id or not reviewer_id.strip():
            raise MissingReviewerError(
                "reviewer_id must be provided. All new labels entering the "
                "label lake require a licensed geologist sign-off. Automated "
                "or anonymous submissions are not permitted."
            )

        csv_path = Path(new_drillhole_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Drillhole CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = {"assay_id", "easting", "northing", "label"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Drillhole CSV is missing required columns: {missing}"
            )

        # Validate label values — only P (prospect), RN (regolith-negative),
        # U (unknown/inconclusive) are permitted.
        valid_labels = {"P", "RN", "U"}
        invalid_labels = set(df["label"].unique()) - valid_labels
        if invalid_labels:
            raise ValueError(
                f"Invalid label values found: {invalid_labels}. "
                f"Only {valid_labels} are permitted."
            )

        # Stamp with governance metadata.
        utc_now = datetime.now(tz=timezone.utc).isoformat()
        df["reviewer_id"] = reviewer_id.strip()
        df["label_source"] = "field_verified"
        df["timestamp"] = utc_now

        # Append-only: write a new shard with a unique timestamp suffix.
        shard_name = (
            f"lake_shard_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.parquet"
        )
        shard_path = self.label_lake_dir / shard_name
        df.to_parquet(shard_path, index=False)

        logger.info(
            "New labels appended | shard=%s | n=%d | reviewer=%s | "
            "label_dist=%s",
            shard_name,
            len(df),
            reviewer_id,
            df["label"].value_counts().to_dict(),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def retrain(
        self,
        feature_stack_path: str,
        version_tag: str,
    ) -> None:
        """Retrain the model from scratch on the full validated label lake.

        Always performs a *complete* retrain — no warm-starting from prior
        checkpoints. This prevents silent catastrophic forgetting and ensures
        the model's decision boundary reflects the current label lake in its
        entirety.

        Steps:
            1. Load and validate the label lake (feedback-loop guard fires here).
            2. Load the feature stack raster and extract labelled pixel values.
            3. Compute SHA-256 hash of the concatenated training data for audit.
            4. Run spatially-blocked k-fold cross-validation; log per-fold metrics.
            5. Train final model on all available labels.
            6. Persist model artefacts, metrics, and the training data hash to
               ``{model_dir}/{version_tag}/``.

        Args:
            feature_stack_path: Path to the multi-band GeoTIFF feature stack
                (output of the feature engineering stage). Bands must be aligned
                to the label coordinate system.
            version_tag: Human-readable version string, e.g. ``"v2"``,
                ``"2026-Q1"``. Used as the versioned subdirectory name.

        Returns:
            None. All artefacts are written to disk.

        Raises:
            FeedbackLoopError: Propagated from :meth:`load_label_lake`.
            FileNotFoundError: If ``feature_stack_path`` does not exist.
        """
        import rasterio
        from rasterio.transform import rowcol

        logger.info("=== Retrain start | version=%s ===", version_tag)

        # 1. Load validated label lake.
        lake = self.load_label_lake()

        # 2. Load feature stack and sample pixel values at label coordinates.
        feat_path = Path(feature_stack_path)
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature stack not found: {feat_path}")

        with rasterio.open(feat_path) as src:
            n_bands = src.count
            # Convert easting/northing to row/col pixel indices.
            rows, cols = rowcol(
                src.transform,
                xs=lake["easting"].values,
                ys=lake["northing"].values,
            )
            # Clip to valid raster extent.
            valid_mask = (
                (rows >= 0) & (rows < src.height) &
                (cols >= 0) & (cols < src.width)
            )
            if valid_mask.sum() < len(lake):
                n_oob = (~valid_mask).sum()
                logger.warning(
                    "%d label point(s) fall outside the feature stack extent "
                    "and will be dropped.",
                    n_oob,
                )
            lake = lake[valid_mask].reset_index(drop=True)
            rows = rows[valid_mask]
            cols = cols[valid_mask]

            # Read all band values at label pixel locations.
            feature_matrix = np.stack(
                [src.read(b + 1)[rows, cols] for b in range(n_bands)],
                axis=1,
            ).astype(np.float32)

        # Binary label: P=1 (prospective), RN/U=0 with PU mining when unlabeled rows exist
        is_p = (lake["label"] == "P").to_numpy()
        is_rn = (lake["label"] == "RN").to_numpy()
        is_u = (lake["label"] == "U").to_numpy()
        if is_u.any() and is_p.any():
            from src.labels.pu_negatives import step1_isolation_forest, step2_select_reliable_negatives
            col_medians = np.nanmedian(feature_matrix, axis=0)
            col_medians = np.nan_to_num(col_medians, nan=0.0)
            inds = np.where(np.isnan(feature_matrix))
            X_clean = feature_matrix.copy()
            X_clean[inds] = np.take(col_medians, inds[1])
            u_feat = X_clean[is_u]
            p_feat = X_clean[is_p]
            anom = step1_isolation_forest(p_feat, u_feat, contamination=0.05, random_state=42)
            rn_sub = step2_select_reliable_negatives(u_feat, anom, percentile_threshold=10.0)
            u_indices = np.where(is_u)[0]
            pseudo_rn_indices = set(u_indices[rn_sub])
            keep_mask = np.array([i in pseudo_rn_indices or is_p[i] or is_rn[i] for i in range(len(lake))])
            lake = lake[keep_mask].reset_index(drop=True)
            feature_matrix = feature_matrix[keep_mask]
        binary_labels = (lake["label"] == "P").astype(int).values

        label_counts = {
            "P": int((lake["label"] == "P").sum()),
            "RN": int((lake["label"] == "RN").sum()),
            "U": int((lake["label"] == "U").sum()),
            "total": len(lake),
        }
        logger.info("Training label counts: %s", label_counts)

        # 3. Compute SHA-256 hash of training data for reproducibility audit.
        hasher = hashlib.sha256()
        hasher.update(feature_matrix.tobytes())
        hasher.update(binary_labels.tobytes())
        data_hash = hasher.hexdigest()
        logger.info("Training data SHA-256: %s", data_hash)

        # 4. Spatially-blocked cross-validation.
        cv_metrics = self._run_spatial_cv(
            feature_matrix, binary_labels, lake
        )

        # 5. Train final model on all data.
        model = self._build_model()
        model.fit(feature_matrix, binary_labels)

        # 6. Persist artefacts.
        version_dir = self.model_dir / version_tag
        version_dir.mkdir(parents=True, exist_ok=True)

        # Persist model (joblib for sklearn-compatible, else native).
        self._save_model(model, version_dir)

        # Write training metadata JSON.
        metadata = {
            "version_tag": version_tag,
            "trained_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "feature_stack_path": str(feat_path),
            "n_features": n_bands,
            "training_data_sha256": data_hash,
            "label_counts": label_counts,
            "cv_metrics": cv_metrics,
        }
        meta_path = version_dir / "training_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "=== Retrain complete | version=%s | artefacts=%s ===",
            version_tag,
            version_dir,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_vs_predecessor(
        self,
        v_new_tag: str,
        v_old_tag: str,
        drillholes_from_v_old: pd.DataFrame,
    ) -> dict:
        """Compare a new model version against its predecessor.

        Evaluates both versions on the subset of drillholes that were *collected
        or tested during the v_old deployment period*. This provides an unbiased
        comparison on data that was **not** available when v_old was trained,
        simulating real prospecting workflow performance.

        Args:
            v_new_tag: Version tag of the candidate new model
                (must exist in ``self.model_dir``).
            v_old_tag: Version tag of the predecessor model to compare against.
            drillholes_from_v_old: DataFrame of drillholes collected/tested
                during the ``v_old`` deployment period. Must contain columns
                ``["easting", "northing", "label"]`` and a ``"features"``
                column (np.ndarray) or pre-extracted feature matrix column
                ``"feature_vector"``.

        Returns:
            dict with keys:
                - ``"v_new_pr_auc"``: PR-AUC of the new model.
                - ``"v_old_pr_auc"``: PR-AUC of the old model.
                - ``"delta_pr_auc"``: v_new − v_old (positive = improvement).
                - ``"v_new_precision_at_50pct_recall"``: precision at 50% recall
                  for new model.
                - ``"v_old_precision_at_50pct_recall"``: precision at 50% recall
                  for old model.
                - ``"n_eval_drillholes"``: number of evaluation drillholes.
                - ``"recommendation"``: ``"promote"`` or ``"keep_old"`` based on
                  PR-AUC delta.

        Raises:
            FileNotFoundError: If either model version directory does not exist.
        """
        for tag in (v_new_tag, v_old_tag):
            vdir = self.model_dir / tag
            if not vdir.exists():
                raise FileNotFoundError(
                    f"Model version directory not found: {vdir}"
                )

        model_new = self._load_model(self.model_dir / v_new_tag)
        model_old = self._load_model(self.model_dir / v_old_tag)

        # Extract feature matrix and binary labels.
        X = self._extract_feature_matrix(drillholes_from_v_old)
        y = (drillholes_from_v_old["label"] == "P").astype(int).values

        prob_new = model_new.predict_proba(X)[:, 1]
        prob_old = model_old.predict_proba(X)[:, 1]

        pr_auc_new = float(average_precision_score(y, prob_new))
        pr_auc_old = float(average_precision_score(y, prob_old))

        # Precision at 50 % recall.
        def _prec_at_recall(y_true: np.ndarray, probs: np.ndarray, target: float = 0.50) -> float:
            """Return precision at the operating point closest to target recall."""
            prec, rec, _ = precision_recall_curve(y_true, probs)
            idx = np.argmin(np.abs(rec - target))
            return float(prec[idx])

        result = {
            "v_new_pr_auc": pr_auc_new,
            "v_old_pr_auc": pr_auc_old,
            "delta_pr_auc": pr_auc_new - pr_auc_old,
            "v_new_precision_at_50pct_recall": _prec_at_recall(y, prob_new),
            "v_old_precision_at_50pct_recall": _prec_at_recall(y, prob_old),
            "n_eval_drillholes": len(drillholes_from_v_old),
            "recommendation": "promote" if pr_auc_new > pr_auc_old else "keep_old",
        }

        logger.info(
            "Model comparison | %s PR-AUC=%.4f | %s PR-AUC=%.4f | delta=%.4f | %s",
            v_new_tag, pr_auc_new,
            v_old_tag, pr_auc_old,
            result["delta_pr_auc"],
            result["recommendation"].upper(),
        )
        return result

    # ------------------------------------------------------------------
    # Changelog
    # ------------------------------------------------------------------

    def generate_changelog(
        self,
        v_old_tag: str,
        v_new_tag: str,
    ) -> str:
        """Produce a human-readable markdown changelog between two model versions.

        Reads the ``training_metadata.json`` artefacts for both versions and
        computes deltas in label counts, PR-AUC, and surface coverage.
        The resulting markdown is suitable for inclusion in a geologist review
        report or a Git commit message.

        Args:
            v_old_tag: Version tag of the predecessor model.
            v_new_tag: Version tag of the candidate new model.

        Returns:
            A multi-line markdown string describing changes between the two
            versions. Also writes the changelog to
            ``{model_dir}/{v_new_tag}/CHANGELOG.md``.

        Raises:
            FileNotFoundError: If either version directory or its
                ``training_metadata.json`` does not exist.
        """
        def _load_meta(tag: str) -> dict:
            meta_path = self.model_dir / tag / "training_metadata.json"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"training_metadata.json not found for version '{tag}': {meta_path}"
                )
            with open(meta_path) as f:
                return json.load(f)

        meta_old = _load_meta(v_old_tag)
        meta_new = _load_meta(v_new_tag)

        lc_old = meta_old.get("label_counts", {})
        lc_new = meta_new.get("label_counts", {})

        def _delta(key: str) -> str:
            old_val = lc_old.get(key, 0)
            new_val = lc_new.get(key, 0)
            d = new_val - old_val
            sign = "+" if d >= 0 else ""
            return f"{new_val} ({sign}{d})"

        cv_old = meta_old.get("cv_metrics", {})
        cv_new = meta_new.get("cv_metrics", {})
        pr_auc_old = cv_old.get("mean_pr_auc", float("nan"))
        pr_auc_new = cv_new.get("mean_pr_auc", float("nan"))
        pr_auc_delta = pr_auc_new - pr_auc_old if not (
            np.isnan(pr_auc_old) or np.isnan(pr_auc_new)
        ) else float("nan")

        trained_new = meta_new.get("trained_at_utc", "unknown")
        data_hash_new = meta_new.get("training_data_sha256", "N/A")

        lines = [
            f"# Model Changelog: `{v_old_tag}` → `{v_new_tag}`",
            "",
            f"**Generated:** {datetime.now(tz=timezone.utc).isoformat()}  ",
            f"**New version trained at:** {trained_new}  ",
            f"**Training data SHA-256:** `{data_hash_new}`",
            "",
            "---",
            "",
            "## Label Lake Changes",
            "",
            "| Category | Count (Δ from predecessor) |",
            "|----------|---------------------------|",
            f"| **P** (Prospective)      | {_delta('P')}  |",
            f"| **RN** (Regolith-Neg.)   | {_delta('RN')} |",
            f"| **U** (Unknown)          | {_delta('U')}  |",
            f"| **Total**                | {_delta('total')} |",
            "",
            "---",
            "",
            "## Model Performance",
            "",
            "| Metric        | Old (`{old}`) | New (`{new}`) | Δ |".format(
                old=v_old_tag, new=v_new_tag
            ),
            "|---------------|---------------|---------------|---|",
            (
                f"| Spatial CV PR-AUC | {pr_auc_old:.4f} | {pr_auc_new:.4f} | "
                f"{pr_auc_delta:+.4f} |"
                if not np.isnan(pr_auc_delta)
                else f"| Spatial CV PR-AUC | {pr_auc_old} | {pr_auc_new} | N/A |"
            ),
            "",
            "---",
            "",
            "## Version Summary",
            "",
            f"- **Feature stack:** `{meta_new.get('feature_stack_path', 'N/A')}`",
            f"- **Number of features:** {meta_new.get('n_features', 'N/A')}",
            "",
            "> **Governance note:** All labels in this version have been",
            "> field-verified by a licensed geologist. No AI predictions",
            "> appear in the label lake (enforced by `FeedbackLoopError` guard).",
            "",
        ]

        changelog_md = "\n".join(lines)

        # Persist alongside model artefacts.
        changelog_path = self.model_dir / v_new_tag / "CHANGELOG.md"
        with open(changelog_path, "w") as f:
            f.write(changelog_md)

        logger.info("Changelog written to %s", changelog_path)
        return changelog_md

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> Any:
        """Instantiate the model specified in ``self.config``.

        Returns:
            A fitted-API-compatible estimator (sklearn or XGBoost).
        """
        model_type = self.config.get("model", {}).get("type", "xgboost")
        if model_type == "xgboost":
            import xgboost as xgb  # type: ignore
            params = dict(
                self.config.get("model", {}).get("params")
                or self.config.get("models", {}).get("baseline", {})
            )
            for k in ("n_bag", "n_bagging_iterations", "type", "uncertainty", "explainability", "hyperparams", "use_label_encoder"):
                params.pop(k, None)
            params.setdefault("eval_metric", "aucpr")
            params.setdefault("random_state", 42)
            return xgb.XGBClassifier(**params)
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            params = self.config.get("model", {}).get("params", {})
            return RandomForestClassifier(random_state=42, **params)
        else:
            raise ValueError(f"Unsupported model type: {model_type!r}")

    def _run_spatial_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lake: pd.DataFrame,
    ) -> dict:
        """Run spatially-blocked k-fold cross-validation.

        Spatial blocking prevents inflated metrics from spatial autocorrelation.
        Labels in the same spatial block are kept together in the same fold,
        ensuring that train and validation sets are geographically separated.

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``.
            y: Binary label vector, shape ``(n_samples,)``.
            lake: Full label lake DataFrame (must contain the spatial block
                column specified in ``config["cv"]["spatial_block_col"]``).

        Returns:
            dict with ``"mean_pr_auc"``, ``"std_pr_auc"``, and
            ``"per_fold_pr_auc"`` keys.
        """
        from sklearn.model_selection import GroupKFold

        n_folds = self.config.get("cv", {}).get("n_folds", 5)
        block_col = self.config.get("cv", {}).get("spatial_block_col", "block_id")

        if block_col not in lake.columns:
            block_size = float(self.config.get("cv", {}).get("block_size_m", 5000.0))
            if "easting" in lake.columns and "northing" in lake.columns:
                bx = np.floor(lake["easting"].to_numpy() / max(block_size, 1e-3)).astype(np.int64)
                by = np.floor(lake["northing"].to_numpy() / max(block_size, 1e-3)).astype(np.int64)
                _, groups = np.unique(bx * 100_003 + by, return_inverse=True)
            else:
                logger.warning(
                    "Spatial block column '%s' and coordinates not found; falling back to fold-stratified groups.",
                    block_col,
                )
                groups = np.arange(len(lake)) % n_folds
            if len(np.unique(groups)) < n_folds:
                groups = np.arange(len(lake)) % n_folds
        else:
            groups = lake[block_col].values

        gkf = GroupKFold(n_splits=n_folds)
        pr_aucs: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(X, y, groups=groups)
        ):
            model_fold = self._build_model()
            model_fold.fit(X[train_idx], y[train_idx])
            probs = model_fold.predict_proba(X[val_idx])[:, 1]
            auc = float(average_precision_score(y[val_idx], probs))
            pr_aucs.append(auc)
            logger.info("  Fold %d/%d — PR-AUC: %.4f", fold_idx + 1, n_folds, auc)

        cv_metrics = {
            "mean_pr_auc": float(np.mean(pr_aucs)),
            "std_pr_auc": float(np.std(pr_aucs)),
            "per_fold_pr_auc": pr_aucs,
            "n_folds": n_folds,
        }
        logger.info(
            "Spatial CV complete | mean PR-AUC=%.4f ± %.4f",
            cv_metrics["mean_pr_auc"],
            cv_metrics["std_pr_auc"],
        )
        return cv_metrics

    def _save_model(self, model: Any, version_dir: Path) -> None:
        """Serialise the trained model to ``{version_dir}/model.joblib``.

        Args:
            model: Trained estimator.
            version_dir: Directory to write the model file to.
        """
        import joblib  # type: ignore
        model_path = version_dir / "model.joblib"
        joblib.dump(model, model_path)
        logger.info("Model saved to %s", model_path)

    def _load_model(self, version_dir: Path) -> Any:
        """Deserialise a model from ``{version_dir}/model.joblib``.

        Args:
            version_dir: Directory containing the saved ``model.joblib``.

        Returns:
            Loaded estimator.

        Raises:
            FileNotFoundError: If ``model.joblib`` is not found.
        """
        import joblib  # type: ignore
        model_path = version_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        logger.info("Loading model from %s", model_path)
        return joblib.load(model_path)

    def _extract_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract a 2D feature matrix from a DataFrame with a 'feature_vector' column.

        Args:
            df: DataFrame containing a ``"feature_vector"`` column where each
                cell holds a 1D numpy array, or individual feature columns.

        Returns:
            numpy array of shape ``(n_samples, n_features)``.
        """
        if "feature_vector" in df.columns:
            return np.vstack(df["feature_vector"].values)
        else:
            # Fall back to all numeric columns except coordinates and labels.
            exclude = {"easting", "northing", "label", "label_source",
                       "reviewer_id", "timestamp", "assay_id"}
            feat_cols = [c for c in df.columns if c not in exclude
                         and pd.api.types.is_numeric_dtype(df[c])]
            return df[feat_cols].values.astype(np.float32)
