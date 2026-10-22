"""XGBoost-based Mineral Prospectivity Mapping (MPM) baseline model.

Rationale for choosing XGBoost as the MPM baseline
----------------------------------------------------
1. **Native NaN handling** — geoscientific raster stacks almost always contain
   masked / no-data pixels (e.g. water bodies, cloud cover, survey gaps).
   XGBoost learns optimal branch directions for missing values at training time,
   avoiding the information loss introduced by imputation.

2. **Non-linear geological thresholds** — mineralisation probability is rarely
   a linear combination of predictors.  Tree ensembles learn step-wise threshold
   responses that model phenomena such as:
   * pH-controlled sulphide stability windows
   * Temperature–pressure envelopes controlling fluid inclusions
   * Structural discontinuities (fault-proximity distance breakpoints)

3. **Efficient on tabular data** — compiled C++ column-block storage means
   XGBoost trains ~10–100× faster than equivalent neural architectures on the
   flat feature tables produced by raster pixel sampling, allowing rapid
   hyperparameter searches and spatial cross-validation loops.

4. **SHAP explainability** — TreeExplainer computes exact Shapley values in
   polynomial time, giving geologists per-pixel feature attributions directly
   usable for prospect prioritisation reports.

5. **Ensemble bagging** — bootstrapped bagging of 10 XGBoost trees provides
   epistemic uncertainty estimates (predictive variance) with minimal extra cost,
   a proxy for data-coverage gaps useful in deciding where to acquire new data.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

logger = logging.getLogger(__name__)


class MPMXGBoost:
    """XGBoost prospectivity classifier with bootstrap-bagged ensemble uncertainty.

    The model wraps ``xgb.XGBClassifier`` in a bootstrap-bagging loop to
    produce both a mean prospectivity score and an epistemic uncertainty map
    (predictive variance across ensemble members).

    Attributes:
        config (dict): Full pipeline configuration dictionary.
        n_bag (int): Number of bagged models in the ensemble.
        ensemble_ (list[xgb.XGBClassifier]): Fitted ensemble members
            (populated after ``fit``).
        feature_names_ (list[str] | None): Feature names inferred from a
            DataFrame input (populated after ``fit``).

    Example:
        >>> cfg = {"models": {"baseline": {"n_estimators": 400, "max_depth": 6,
        ...         "learning_rate": 0.05, "subsample": 0.8,
        ...         "colsample_bytree": 0.8, "min_child_weight": 5,
        ...         "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
        ...         "use_label_encoder": False, "eval_metric": "aucpr",
        ...         "tree_method": "hist", "random_state": 42, "n_bag": 10}}}
        >>> model = MPMXGBoost(cfg)
        >>> model.fit(X_train, y_train)
        >>> scores = model.predict_proba(X_test)
    """

    def __init__(self, config: dict) -> None:
        """Initialise MPMXGBoost from a pipeline configuration dictionary.

        Args:
            config (dict): Pipeline config; must contain the key path
                ``config["models"]["baseline"]`` with at least the XGBoost
                hyperparameters listed in the class docstring.  The optional
                key ``n_bag`` (default 10) controls ensemble size.
        """
        self.config: dict = config
        models_cfg = config.get("models", {})
        baseline_cfg: dict = (
            models_cfg.get("baseline")
            or models_cfg.get("xgboost")
            or {}
        )

        # Number of bootstrap ensemble members (epistemic uncertainty proxy)
        self.n_bag: int = int(
            baseline_cfg.get("n_bag", baseline_cfg.get("n_bagging_iterations", 10))
        )

        # Store hyperparameter block
        hparams: dict = dict(baseline_cfg.get("hyperparams", {}))
        for k, v in baseline_cfg.items():
            if k not in ("hyperparams", "n_bag", "n_bagging_iterations", "type", "uncertainty", "explainability"):
                hparams.setdefault(k, v)
        if hparams.get("missing") == "NaN" or hparams.get("missing") == "nan":
            hparams["missing"] = np.nan
        if hparams.get("scale_pos_weight") == "auto":
            hparams.pop("scale_pos_weight", None)
        # use_label_encoder was deprecated in recent XGBoost releases
        hparams.pop("use_label_encoder", None)
        self._hparams: dict = hparams

        # Populated by fit()
        self.ensemble_: list[xgb.XGBClassifier] = []
        self.feature_names_: Optional[list[str]] = None

        logger.info(
            "MPMXGBoost initialised | n_bag=%d | n_estimators=%s",
            self.n_bag,
            self._hparams.get("n_estimators", "default"),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_single_model(
        self, scale_pos_weight: float, bag_idx: int = 0
    ) -> xgb.XGBClassifier:
        """Construct one XGBClassifier with the configured hyperparameters.

        The ``scale_pos_weight`` argument is computed externally from the
        positive-to-random-negative (P:RN) ratio in the training set and
        passed here so that each bagged member uses the *bag-specific* class
        balance (bootstrapping slightly perturbs the ratio).

        Args:
            scale_pos_weight (float): Ratio of negative to positive samples
                used to up-weight the minority (mineralised) class.  A value
                of ``n_negatives / n_positives`` is the XGBoost convention.
            bag_idx (int): Index of the ensemble bag, used to perturb the
                random state across bootstrap iterations.

        Returns:
            xgb.XGBClassifier: Unfitted classifier ready for training.
        """
        params = dict(self._hparams)  # shallow copy
        params["scale_pos_weight"] = scale_pos_weight
        base_seed = int(params.get("random_state", 42))
        params["random_state"] = base_seed + int(bag_idx)

        # Ensure eval_metric doesn't clash with early stopping API
        params.setdefault("eval_metric", "aucpr")
        params.setdefault("tree_method", "hist")

        clf = xgb.XGBClassifier(**params)
        return clf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        eval_set: Optional[list] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "MPMXGBoost":
        """Train the bagged XGBoost ensemble.

        Trains ``self.n_bag`` XGBoost classifiers, each on an 80 % bootstrap
        sample of the training data drawn *with replacement*.  The slight
        perturbation in both data coverage and, for XGBoost's internal
        subsampling, in random seeds ensures diversity across ensemble members
        — the primary source of the epistemic uncertainty signal.

        Class imbalance handling:
            ``scale_pos_weight`` = (# negatives) / (# positives) in each
            bootstrap bag, following the XGBoost recommendation for
            imbalanced binary classification.  Geological training sets
            typically have a 50:1 – 500:1 negative-to-positive ratio.

        Args:
            X (np.ndarray | pd.DataFrame): Feature matrix of shape
                ``(N, n_features)``.  If a DataFrame is supplied, column names
                are stored in ``self.feature_names_``.
            y (np.ndarray): Binary labels of shape ``(N,)``; 1 = mineralised
                (known deposit / occurrence), 0 = random negative.
            eval_set (list | None): Optional list of ``(X_val, y_val)`` tuples
                passed through to ``xgb.fit`` for verbose loss monitoring.
                Not used for early stopping in the baseline to keep evaluation
                leakage-free.
            sample_weight (np.ndarray | None): Optional inverse-propensity
                sample weights $w_i = 1 / e(x_i)$ of shape ``(N,)`` from Pass 2
                to penalise easy-access discoveries and force learning of true
                basement signatures under cover.

        Returns:
            MPMXGBoost: self, to allow method chaining.
        """
        # Capture feature names from DataFrame columns
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            self.feature_names_ = None

        y_arr = np.asarray(y, dtype=np.int32)
        n_samples = X_arr.shape[0]
        w_arr: Optional[np.ndarray] = None
        if sample_weight is not None:
            w_arr = np.asarray(sample_weight, dtype=np.float64)
            if w_arr.shape[0] != n_samples:
                raise ValueError(
                    f"sample_weight length ({w_arr.shape[0]}) != X rows ({n_samples})"
                )

        bag_size = int(0.80 * n_samples)

        self.ensemble_ = []

        rng = np.random.default_rng(
            self._hparams.get("random_state", 42)
        )

        for bag_idx in range(self.n_bag):
            # Bootstrap resample (with replacement, 80 % of training set)
            idx = rng.choice(n_samples, size=bag_size, replace=True)
            X_bag = X_arr[idx]
            y_bag = y_arr[idx]

            # Compute per-bag class balance
            n_pos = int(y_bag.sum())
            n_neg = bag_size - n_pos
            if n_pos == 0:
                logger.warning(
                    "Bag %d contains no positive samples — skipping.", bag_idx
                )
                continue

            # Avoid double-weighting when class-balanced sample_weight is already supplied
            spw = 1.0 if w_arr is not None else (n_neg / n_pos)
            logger.debug(
                "Bag %d/%d | n_pos=%d | n_neg=%d | scale_pos_weight=%.2f",
                bag_idx + 1,
                self.n_bag,
                n_pos,
                n_neg,
                spw,
            )

            clf = self._build_single_model(scale_pos_weight=spw, bag_idx=bag_idx)

            fit_kwargs: dict = {}
            if eval_set is not None:
                fit_kwargs["eval_set"] = eval_set
                fit_kwargs["verbose"] = False
            if w_arr is not None:
                fit_kwargs["sample_weight"] = w_arr[idx]

            clf.fit(X_bag, y_bag, **fit_kwargs)
            self.ensemble_.append(clf)

        logger.info(
            "MPMXGBoost ensemble fitted | %d/%d bags completed",
            len(self.ensemble_),
            self.n_bag,
        )
        return self

    def predict_proba(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Predict mean prospectivity probability across ensemble members.

        Averages the positive-class (class 1) probability from all ``n_bag``
        ensemble members.  This ensemble mean is a better-calibrated estimate
        than any single member because variance from individual tree noise is
        averaged out.

        Args:
            X (np.ndarray | pd.DataFrame): Feature matrix of shape
                ``(N, n_features)``.

        Returns:
            np.ndarray: Mean prospectivity scores of shape ``(N,)`` in [0, 1].

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        if not self.ensemble_:
            raise RuntimeError("Call fit() before predict_proba().")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        # Stack predictions: shape (n_bag, N)
        preds = np.stack(
            [clf.predict_proba(X_arr)[:, 1] for clf in self.ensemble_],
            axis=0,
        )
        return preds.mean(axis=0)  # shape (N,)

    def predict_uncertainty(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Compute epistemic uncertainty as predictive variance across the ensemble.

        High variance regions correspond to areas where bagged models disagree —
        typically caused by sparse training coverage (data uncertainty) or
        conflicting feature signals (model uncertainty).  In the MPM context
        these are the most important regions to target for follow-up field data
        acquisition rather than immediate drilling.

        Args:
            X (np.ndarray | pd.DataFrame): Feature matrix of shape
                ``(N, n_features)``.

        Returns:
            np.ndarray: Per-pixel variance of shape ``(N,)`` in [0, 0.25].
                Maximum possible variance for a Bernoulli variable is 0.25
                (p = 0.5).

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        if not self.ensemble_:
            raise RuntimeError("Call fit() before predict_uncertainty().")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        # Stack predictions: shape (n_bag, N)
        preds = np.stack(
            [clf.predict_proba(X_arr)[:, 1] for clf in self.ensemble_],
            axis=0,
        )
        return preds.var(axis=0)  # shape (N,)

    def compute_shap(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Compute SHAP values using the first ensemble member as a representative.

        Uses ``shap.TreeExplainer`` which runs in O(T·D·N) time (T = trees,
        D = max depth, N = samples) — exact, not sampled.  We use ensemble_[0]
        as a representative member because:
        (a) SHAP values are additive across models but the first member was
            trained on 80 % of data with the same hyperparameters, so its
            feature importance ordering is representative of the full ensemble;
        (b) computing SHAP for all 10 ensemble members and averaging is
            10× more expensive and rarely changes the top-5 feature ranking.

        For production use, call this on a subsample (e.g. 5 000 pixels) and
        extrapolate global importance to the full raster.

        Args:
            X (np.ndarray | pd.DataFrame): Feature matrix of shape
                ``(N, n_features)``.  Pass a DataFrame to get named features
                in the returned explainer object.

        Returns:
            np.ndarray: SHAP value matrix of shape ``(N, n_features)``.
                Positive values increase predicted probability; negative values
                decrease it.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        if not self.ensemble_:
            raise RuntimeError("Call fit() before compute_shap().")

        representative_model = self.ensemble_[0]
        explainer = shap.TreeExplainer(representative_model)
        shap_values = explainer.shap_values(X)

        # For binary XGBClassifier, shap_values may be a list [neg_class, pos_class]
        # We want the positive (mineralised) class SHAP values.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return np.asarray(shap_values)  # shape (N, n_features)

    def save(self, path: str) -> None:
        """Persist the fitted ensemble to disk using joblib compression.

        Saves the list of fitted ``xgb.XGBClassifier`` objects together with
        auxiliary attributes (feature names, hyperparameter block) to allow
        exact reproduction of inference without re-training.

        Args:
            path (str): Destination file path (e.g. ``"models/xgb_ensemble.joblib"``).
                Parent directory must already exist.
        """
        payload = {
            "ensemble": self.ensemble_,
            "feature_names": self.feature_names_,
            "hparams": self._hparams,
            "n_bag": self.n_bag,
        }
        joblib.dump(payload, path, compress=3)
        logger.info("MPMXGBoost ensemble saved → %s", path)

    def load(self, path: str) -> "MPMXGBoost":
        """Load a previously saved ensemble from disk.

        Restores the fitted ensemble list and auxiliary attributes from a
        joblib file created by ``save()``.

        Args:
            path (str): Path to the ``.joblib`` file written by ``save()``.

        Returns:
            MPMXGBoost: self, to allow method chaining.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        payload: dict = joblib.load(path)
        self.ensemble_ = payload["ensemble"]
        self.feature_names_ = payload.get("feature_names")
        self._hparams = payload.get("hparams", self._hparams)
        self.n_bag = payload.get("n_bag", self.n_bag)
        logger.info(
            "MPMXGBoost ensemble loaded ← %s | %d members",
            path,
            len(self.ensemble_),
        )
        return self
