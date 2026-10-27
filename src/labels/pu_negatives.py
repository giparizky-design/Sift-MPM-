"""Two-step PU (Positive-Unlabeled) reliable negative identification.

PU Learning Background
----------------------
Standard supervised learning requires labelled positives *and* negatives.
In MPM the negative class is never directly observed – we only know where
deposits exist (P) and where they have *not yet been found* (U).  Treating
all U as negative introduces systematic bias because U contains undiscovered
deposits (Elkan & Noto 2008).

Two-step PU approach (Liu et al. 2003; Mordelet & Vert 2014)
-------------------------------------------------------------
Step 1 – Identify Reliable Negatives from U:
    Train an IsolationForest anomaly detector on the feature matrix of P
    and U cells.  Score every U cell.  U cells with the *lowest* anomaly
    score (i.e., most similar to the bulk background) are selected as
    Reliable Negatives (RN).  The IsolationForest is preferred over naive
    distance methods because it handles high-dimensional, mixed-scale
    geophysical feature spaces naturally without a distance metric.

Step 2 – PU Bagging (Mordelet & Vert 2014):
    Train an ensemble of base classifiers.  Each base model sees all P
    plus a random subsample of U (treated provisionally as N).  The
    ensemble prediction is the average probability across all base models.
    This averaging implicitly handles the noise in the negative label:
    any true positive masquerading as U will be misclassified in some
    models but not all; averaging suppresses this noise.

References
----------
Liu, B., Dai, Y., Li, X., Lee, W.S., & Yu, P.S. (2003). Building text
    classifiers using positive and unlabeled examples. ICDM.
Mordelet, F., & Vert, J.-P. (2014). A bagging SVM to learn from positive
    and unlabeled examples. Pattern Recognition Letters, 37, 201-209.
Elkan, C., & Noto, K. (2008). Learning classifiers from only positive and
    unlabeled data. KDD.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.utils import check_random_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 – IsolationForest anomaly scoring
# ---------------------------------------------------------------------------


def step1_isolation_forest(
    X_positive: np.ndarray,
    X_unlabeled: np.ndarray,
    contamination: float = 0.05,
    random_state: int = 42,
) -> np.ndarray:
    """Score unlabeled cells by their departure from background normality.

    An IsolationForest is trained on ``X_unlabeled`` only.  The
    ``contamination`` parameter tells the forest what fraction of U cells
    it should treat as anomalous (here: deposit-like).  After training,
    the decision function is evaluated on ``X_unlabeled``; lower (more
    negative) scores indicate cells that are *harder* to isolate (more
    similar to the background) – these are candidates for RN.

    Note: ``X_positive`` is accepted as a parameter for future extensions
    (e.g., computing score thresholds relative to P distribution) and for
    API consistency.  In the basic implementation it is not used during
    IsolationForest training because the forest should learn the background
    distribution, not the P distribution.

    Args:
        X_positive:    Feature matrix for P cells, shape (n_p, n_features).
                       Currently used only for logging/diagnostics.
        X_unlabeled:   Feature matrix for U cells, shape (n_u, n_features).
        contamination: Expected fraction of anomalous samples in U.
                       Should reflect the *expected prevalence* of
                       undiscovered deposits in the study area.
                       Defaults to 0.05 (5 %).
        random_state:  Random seed for reproducibility.

    Returns:
         scores: 1-D array of shape (n_u,).  Anomaly score where **higher =
                 more anomalous** (negated sklearn ``decision_function``) and
                 **lower = more normal (background-like)**, i.e. the best
                 Reliable-Negative candidates.
    """
    if X_unlabeled.ndim != 2:
        raise ValueError(
            f"X_unlabeled must be 2-D, got shape {X_unlabeled.shape}"
        )
    if X_positive.ndim != 2:
        raise ValueError(
            f"X_positive must be 2-D, got shape {X_positive.shape}"
        )
    if not (0.0 < contamination < 0.5):
        raise ValueError(
            f"contamination must be in (0, 0.5), got {contamination}"
        )

    logger.info(
        "Step 1 IsolationForest: fitting on %d unlabeled samples, "
        "%d features, contamination=%.3f",
        X_unlabeled.shape[0], X_unlabeled.shape[1], contamination,
    )

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_unlabeled)

    # sklearn's decision_function is HIGHER for inliers (normal) and NEGATIVE for outliers.
    # Our contract (see docstring and step2) is the opposite: HIGHER = MORE ANOMALOUS
    # (deposit-like), LOWER = MORE NORMAL (background, safe to treat as reliable negative).
    # Negate so that selecting low scores in step 2 picks the most background-like cells.
    scores: np.ndarray = -iso.decision_function(X_unlabeled)

    logger.info(
        "Anomaly scores – min: %.4f  median: %.4f  max: %.4f",
        float(scores.min()), float(np.median(scores)), float(scores.max()),
    )
    return scores


# ---------------------------------------------------------------------------
# Step 2 – Select reliable negatives by percentile thresholding
# ---------------------------------------------------------------------------


def step2_select_reliable_negatives(
    X_unlabeled: np.ndarray,
    scores: np.ndarray,
    percentile_threshold: float = 10.0,
) -> np.ndarray:
    """Select the most background-like U cells as Reliable Negatives.

    U cells whose anomaly score falls at or below ``percentile_threshold``
    of the score distribution are most similar to the background and least
    likely to be undiscovered deposits.  These are promoted to the RN set
    for Step 2 classifier training.

    Choosing a low percentile (e.g., 10th) is conservative: we only select
    cells that are unambiguously background-like, reducing the risk of
    mislabelling true positives.  A higher percentile yields more RN
    samples at the cost of higher contamination risk.

    Args:
        X_unlabeled:          Feature matrix for U cells, shape (n_u, n_features).
                              Only used for shape validation; the mask returned
                              indexes into this matrix.
        scores:               Anomaly scores from :func:`step1_isolation_forest`,
                              shape (n_u,).  Lower = more normal.
        percentile_threshold: Percentile of the score distribution below which
                              U cells are selected as RN.  Defaults to 10.0
                              (bottom 10 % of scores = most normal).

    Returns:
        Boolean mask of shape (n_u,).  ``True`` indicates the corresponding
        U cell is selected as a Reliable Negative.
    """
    if scores.shape[0] != X_unlabeled.shape[0]:
        raise ValueError(
            f"scores length ({scores.shape[0]}) must match "
            f"X_unlabeled rows ({X_unlabeled.shape[0]})"
        )
    if not (0.0 < percentile_threshold < 100.0):
        raise ValueError(
            f"percentile_threshold must be in (0, 100), got {percentile_threshold}"
        )

    threshold_value: float = float(np.percentile(scores, percentile_threshold))
    rn_mask: np.ndarray = scores <= threshold_value

    logger.info(
        "Step 2 RN selection: score threshold (%.1f pct) = %.4f | "
        "selected %d / %d U cells as RN",
        percentile_threshold,
        threshold_value,
        int(rn_mask.sum()),
        len(scores),
    )
    return rn_mask


# ---------------------------------------------------------------------------
# Orchestrator: build full PU training set
# ---------------------------------------------------------------------------


def build_pu_training_set(
    X_p: np.ndarray,
    X_u: np.ndarray,
    y_labels: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Orchestrate two-step PU to produce a clean (X_train, y_train) set.

    Workflow
    --------
    1. Run :func:`step1_isolation_forest` to score all U cells.
    2. Run :func:`step2_select_reliable_negatives` to select RN from U.
    3. Concatenate P (y=1) and new RN (y=0) into a labelled training set.

    The resulting training set is balanced (or imbalanced) depending on
    ``config['balance']`` and ``config['random_state']``.

    Args:
        X_p:       Feature matrix for P cells, shape (n_p, n_features).
        X_u:       Feature matrix for U cells, shape (n_u, n_features).
        y_labels:  Label array for all P+U cells (1 for P, -1 for U).
                   Shape (n_p + n_u,).  Used only for bookkeeping; the
                   function constructs new y_train internally.
        config:    Configuration dictionary supporting keys:

                   - ``contamination`` (float, default 0.05):
                     Passed to :func:`step1_isolation_forest`.
                   - ``percentile_threshold`` (float, default 10.0):
                     Passed to :func:`step2_select_reliable_negatives`.
                   - ``random_state`` (int, default 42):
                     Seed for reproducibility.
                   - ``balance`` (bool, default False):
                     If True, randomly downsample RN to match |P|.

    Returns:
        Tuple ``(X_train, y_train)`` where:
            - X_train: shape (n_p + n_rn, n_features)
            - y_train: shape (n_p + n_rn,), values in {0, 1}
    """
    contamination: float = float(config.get("contamination", 0.05))
    percentile_threshold: float = float(config.get("percentile_threshold", 10.0))
    random_state: int = int(config.get("random_state", 42))
    balance: bool = bool(config.get("balance", False))

    rng = check_random_state(random_state)

    # Step 1: score U cells
    scores = step1_isolation_forest(
        X_positive=X_p,
        X_unlabeled=X_u,
        contamination=contamination,
        random_state=random_state,
    )

    # Step 2: select RN from U
    rn_mask = step2_select_reliable_negatives(
        X_unlabeled=X_u,
        scores=scores,
        percentile_threshold=percentile_threshold,
    )
    X_rn = X_u[rn_mask]

    # Optional: downsample RN to match |P| for class balance
    if balance and len(X_rn) > len(X_p):
        chosen = rng.choice(len(X_rn), size=len(X_p), replace=False)
        X_rn = X_rn[chosen]
        logger.info(
            "build_pu_training_set: downsampled RN from %d to %d to balance with P.",
            int(rn_mask.sum()), len(X_p),
        )

    # Concatenate
    X_train = np.vstack([X_p, X_rn])
    y_train = np.concatenate([
        np.ones(len(X_p), dtype=np.int8),
        np.zeros(len(X_rn), dtype=np.int8),
    ])

    logger.info(
        "PU training set built: %d P + %d RN = %d samples, %d features",
        len(X_p), len(X_rn), len(X_train), X_train.shape[1],
    )
    return X_train, y_train


# ---------------------------------------------------------------------------
# PU Bagging ensemble
# ---------------------------------------------------------------------------


def pu_bagging_wrapper(
    base_model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 10,
    subsample_negative_ratio: float = 1.0,
    random_state: int = 42,
) -> list[Any]:
    """PU bagging ensemble following Mordelet & Vert (2014).

    Algorithm
    ---------
    For each of ``n_estimators`` iterations:

    1. Take *all* P samples (y == 1).
    2. Draw a random subsample of U samples (y == -1 or y == 0) of size
       ``subsample_negative_ratio × |P|``.  Using a subsample prevents the
       ensemble from seeing any single U cell too often, which would bias
       the model towards treating U as reliably negative.
    3. Train a clone of ``base_model`` on P + subsample-U, treating U as
       negative (y = 0).
    4. Accumulate the fitted model in a list.

    Ensemble prediction is computed downstream by averaging
    ``model.predict_proba(X)[:, 1]`` across all models.

    Args:
        base_model:               An unfitted scikit-learn compatible estimator
                                  with a ``fit(X, y)`` method and a
                                  ``predict_proba(X)`` method.
        X:                        Full feature matrix (P + U), shape
                                  (n_samples, n_features).
        y:                        Label array: 1 for P, 0 or -1 for U.
                                  Shape (n_samples,).
        n_estimators:             Number of base models to train.
        subsample_negative_ratio: Ratio of U samples to draw per estimator
                                  relative to |P|.  1.0 means draw |P| U
                                  samples; 2.0 means draw 2×|P| U samples.
                                  Higher ratios increase recall at the cost
                                  of precision.
        random_state:             Base random seed.  Each estimator uses
                                  ``random_state + i`` for reproducibility.

    Returns:
        List of ``n_estimators`` fitted base model clones.

    Raises:
        ValueError: If no P or U samples are found in ``y``.
    """
    from sklearn.base import clone as sklearn_clone

    if not (0.0 < subsample_negative_ratio):
        raise ValueError(
            f"subsample_negative_ratio must be > 0, got {subsample_negative_ratio}"
        )

    p_idx = np.where(y == 1)[0]
    u_idx = np.where(y != 1)[0]  # covers y==0 and y==-1

    if len(p_idx) == 0:
        raise ValueError("No positive samples (y == 1) found in y.")
    if len(u_idx) == 0:
        raise ValueError("No unlabeled/negative samples found in y.")

    n_draw = max(1, int(round(len(p_idx) * subsample_negative_ratio)))
    n_draw = min(n_draw, len(u_idx))  # cannot exceed available U samples

    logger.info(
        "PU bagging: %d estimators | |P|=%d | |U|=%d | drawing %d U per estimator",
        n_estimators, len(p_idx), len(u_idx), n_draw,
    )

    fitted_models: list[Any] = []
    X_p = X[p_idx]
    X_u = X[u_idx]

    for i in range(n_estimators):
        rng_i = check_random_state(random_state + i)

        # Random subsample of U (without replacement when possible)
        replace = n_draw > len(u_idx)
        chosen_u = rng_i.choice(len(u_idx), size=n_draw, replace=replace)
        X_u_sub = X_u[chosen_u]

        # Build mini training set: P (y=1) + U subsample (y=0)
        X_iter = np.vstack([X_p, X_u_sub])
        y_iter = np.concatenate([
            np.ones(len(X_p), dtype=np.int8),
            np.zeros(n_draw, dtype=np.int8),
        ])

        # Clone and fit base model
        model_i = sklearn_clone(base_model)

        # Propagate random_state if the model supports it
        if hasattr(model_i, "random_state"):
            model_i.random_state = random_state + i

        model_i.fit(X_iter, y_iter)
        fitted_models.append(model_i)

        logger.debug(
            "PU bagging estimator %d/%d fitted on %d samples.",
            i + 1, n_estimators, len(X_iter),
        )

    logger.info("PU bagging complete: %d estimators fitted.", len(fitted_models))
    return fitted_models


# ---------------------------------------------------------------------------
# Pass 2: Selection-Bias Weighting (SAR Observation Propensity)
# ---------------------------------------------------------------------------


def fit_exploration_propensity(
    accessibility_features: np.ndarray,
    explored_mask: np.ndarray,
    min_propensity: float = 0.05,
    max_propensity: float = 0.99,
    random_state: int = 42,
) -> np.ndarray:
    """Fit an observation propensity model $e(x) = P(\\text{explored} \\mid x)$.

    Historical mineral discoveries cluster near roads, railways, and exposed
    outcrops/hilltops due to geologist foot-access bias. Under the Selected At
    Random (SAR) assumption conditioned on accessibility covariates (regolith
    cover thickness, outcrop exposure, distance to roads/railways), this
    function estimates the observation propensity $e(x_i)$ for each pixel $x_i$
    using an L2-regularised logistic regression model over standardised
    accessibility covariates.

    To satisfy **Acceptance Criterion 2 (Propensity Stability)**, all estimated
    propensities are strictly bounded away from zero:
    $e(x_i) \\leftarrow \\max(e(x_i), \\text{min\\_propensity})$ (default $0.05$).

    Args:
        accessibility_features: 2-D array of shape ``(N, n_cov)`` containing
            accessibility covariates (e.g. ``[regolith_depth_m, outcrop_exposure,
            dist_to_road_m]``). NaNs are imputed with column medians.
        explored_mask: 1-D boolean or binary array of shape ``(N,)`` where ``1``
            or ``True`` denotes historically explored / drilled / sampled locations.
        min_propensity: Lower bound on $e(x_i)$ to prevent exploding inverse
            propensity weights in XGBoost and U-Net. Must be $> 0$. Defaults to ``0.05``.
        max_propensity: Upper bound on $e(x_i)$. Defaults to ``0.99``.
        random_state: Random seed for solver reproducibility.

    Returns:
        1-D float64 array of bounded observation propensities $e(x_i) \\in
        [\\text{min\\_propensity}, \\text{max\\_propensity}]$ of shape ``(N,)``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if min_propensity <= 0.0 or min_propensity >= 1.0:
        raise ValueError(f"min_propensity must be in (0, 1), got {min_propensity}")

    X_acc = np.asarray(accessibility_features, dtype=np.float64)
    if X_acc.ndim == 1:
        X_acc = X_acc[:, np.newaxis]
    if X_acc.ndim != 2:
        raise ValueError(
            f"accessibility_features must be 2-D, got shape {X_acc.shape}"
        )

    s = (np.asarray(explored_mask) > 0).astype(np.int32).ravel()
    if len(s) != X_acc.shape[0]:
        raise ValueError(
            f"explored_mask length ({len(s)}) != accessibility_features rows ({X_acc.shape[0]})"
        )

    # Median-fill NaNs per column
    col_medians = np.nanmedian(X_acc, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    X_clean = X_acc.copy()
    nan_r, nan_c = np.where(np.isnan(X_clean))
    if len(nan_r) > 0:
        X_clean[nan_r, nan_c] = col_medians[nan_c]

    if len(np.unique(s)) < 2:
        base_rate = float(np.clip(np.mean(s), min_propensity, max_propensity))
        return np.full(len(s), base_rate, dtype=np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    clf = LogisticRegression(C=1.0, max_iter=500, random_state=random_state)
    clf.fit(X_scaled, s)
    e_x = clf.predict_proba(X_scaled)[:, 1]
    e_x_bounded = np.clip(e_x, float(min_propensity), float(max_propensity))

    logger.info(
        "fit_exploration_propensity: e(x) min=%.4f median=%.4f max=%.4f (floor=%.3f)",
        float(e_x_bounded.min()),
        float(np.median(e_x_bounded)),
        float(e_x_bounded.max()),
        min_propensity,
    )
    return e_x_bounded.astype(np.float64)


def compute_inverse_propensity_weights(
    propensity: np.ndarray,
    min_propensity: float = 0.05,
    max_weight: Optional[float] = None,
) -> np.ndarray:
    """Compute clipped inverse propensity sample weights $w_i = 1 / e(x_i)$.

    Penalises easy-access historical discoveries (high $e(x_i) \\Rightarrow$ low
    $w_i$) and up-weights discoveries or background constraints under deep cover
    (low $e(x_i) \\Rightarrow$ high $w_i$), forcing downstream tabular and
    spatial learners to learn true basement signatures rather than road/outcrop
    proximity.

    Args:
        propensity: 1-D array of observation propensities $e(x_i)$.
        min_propensity: Minimum floor applied to $e(x_i)$ before inversion
            (default ``0.05``, giving a maximum weight of ``20.0``).
        max_weight: Optional explicit upper cap on $w_i$. If ``None``, defaults
            to ``1.0 / min_propensity``.

    Returns:
        1-D float64 array of inverse propensity weights $w_i$ of shape ``(N,)``.
    """
    if min_propensity <= 0.0:
        raise ValueError(f"min_propensity must be > 0, got {min_propensity}")

    e_safe = np.maximum(np.asarray(propensity, dtype=np.float64), float(min_propensity))
    weights = 1.0 / e_safe
    cap = float(max_weight) if max_weight is not None else (1.0 / float(min_propensity))
    weights = np.clip(weights, 1.0, cap)
    return weights.astype(np.float64)


# ---------------------------------------------------------------------------
# Pass 2: Pooled Propensity, N x 4 Exclusion Buffers, Asymmetric Weights & OUTMOGC
# ---------------------------------------------------------------------------


def fit_pooled_exploration_propensity(
    accessibility_features: np.ndarray,
    explored_footprint: np.ndarray,
    min_propensity: float = 0.05,
    max_propensity: float = 1.0,
    model_type: str = "logistic",
    random_state: int = 42,
) -> np.ndarray:
    """Fit a single pooled observation propensity model on the exploration footprint.

    Enforces Task 1.3 Rule 1:
      Fits a single logistic or random forest propensity estimator on the pooled
      exploration footprint (N ≈ 1,800) using accessibility covariates (distance to
      navigable rivers, road tracks, coast, and DEM slope); clips predictions to [0.05, 1.0].

    Args:
        accessibility_features: 2-D array of shape (N, n_covariates).
        explored_footprint: 1-D binary array of shape (N,) where 1 denotes explored.
        min_propensity: Floor on observation propensity (default 0.05).
        max_propensity: Upper cap on observation propensity (default 1.0).
        model_type: 'logistic' or 'rf' (Random Forest). Defaults to 'logistic'.
        random_state: Solver random seed.

    Returns:
        1-D float64 array of observation propensities e_hat_logistics(s) in [0.05, 1.0].
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    if min_propensity <= 0.0 or min_propensity >= 1.0:
        raise ValueError(f"min_propensity must be in (0, 1), got {min_propensity}")

    X_acc = np.asarray(accessibility_features, dtype=np.float64)
    if X_acc.ndim == 1:
        X_acc = X_acc[:, np.newaxis]
    if X_acc.ndim != 2:
        raise ValueError(f"accessibility_features must be 2-D, got shape {X_acc.shape}")

    s = (np.asarray(explored_footprint) > 0).astype(np.int32).ravel()
    if len(s) != X_acc.shape[0]:
        raise ValueError(
            f"explored_footprint length ({len(s)}) != accessibility_features rows ({X_acc.shape[0]})"
        )

    # Impute NaNs with column medians
    col_medians = np.nanmedian(X_acc, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    X_clean = X_acc.copy()
    nan_r, nan_c = np.where(np.isnan(X_clean))
    if len(nan_r) > 0:
        X_clean[nan_r, nan_c] = col_medians[nan_c]

    if len(np.unique(s)) < 2:
        base_rate = float(np.clip(np.mean(s), min_propensity, max_propensity))
        return np.full(len(s), base_rate, dtype=np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    if model_type.lower() == "rf":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=random_state)

    clf.fit(X_scaled, s)
    e_raw = clf.predict_proba(X_scaled)[:, 1]
    e_clipped = np.clip(e_raw, float(min_propensity), float(max_propensity))

    logger.info(
        "fit_pooled_exploration_propensity (%s): e(s) min=%.4f median=%.4f max=%.4f",
        model_type,
        float(e_clipped.min()),
        float(np.median(e_clipped)),
        float(e_clipped.max()),
    )
    return e_clipped.astype(np.float64)


def compute_k4_exclusion_mask(
    coords: np.ndarray,
    positive_coords_k: dict[int, np.ndarray],
    labels_k4: Optional[np.ndarray] = None,
    is_geographic: bool = True,
    r_k_star_m: Optional[dict[int, Any]] = None,
) -> np.ndarray:
    """Compute the N x 4 exclusion buffer mask M to eliminate negative halo contamination.

    Enforces Task 1.3 Rule 2:
      For each class k in {0, 1, 2, 3}, sets M[i, k] = 0.0 if cell i has label_k == -1 and lies
      within r_k* of any positive cell in P_k:
        - r_0* = 6.0 km (isotropic, Porphyry Cu-Au)
        - r_1* = 3.5 km (isotropic, Contact Fe-Cu Skarn)
        - r_2* = 6.0 km (isotropic, Epithermal Au-Ag)
        - r_3* = 10.0 km along-strike x 2.0 km across-strike (anisotropic, Meratus NE-SW Ophiolite Lateritic Fe)
      Otherwise M[i, k] = 1.0.

    Args:
        coords: (N, 2) array of cell coordinates ([lon, lat] or projected [easting, northing]).
        positive_coords_k: Dictionary mapping head index k in {0, 1, 2, 3} to (N_k, 2) positive coordinates.
        labels_k4: Optional (N, 4) array of labels in {-1, 1}. If supplied, mask 0.0 applies only to
            unlabeled cells (label == -1); positive cells remain 1.0.
        is_geographic: Whether coords are in decimal degrees (EPSG:4326).
        r_k_star_m: Optional dictionary overriding default buffer radii.

    Returns:
        (N, 4) float64 array M with values in {0.0, 1.0}.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(coords, dtype=np.float64)
    N = xy.shape[0]
    M = np.ones((N, 4), dtype=np.float64)

    if is_geographic:
        mean_lat = float(np.nanmean(xy[:, 1]))
        m_per_deg_lon = 111320.0 * np.cos(np.radians(mean_lat))
        m_per_deg_lat = 110570.0
        metric_xy = np.column_stack([xy[:, 0] * m_per_deg_lon, xy[:, 1] * m_per_deg_lat])
    else:
        metric_xy = xy
        m_per_deg_lon = 1.0
        m_per_deg_lat = 1.0

    tree = cKDTree(metric_xy)

    # Default buffer radii in metres
    default_r = {
        0: 6000.0,   # 6.0 km isotropic
        1: 3500.0,   # 3.5 km isotropic
        2: 6000.0,   # 6.0 km isotropic
        3: (10000.0, 2000.0, 45.0),  # 10.0 km along-strike x 2.0 km across-strike, azimuth 45°
    }
    radii = r_k_star_m if r_k_star_m is not None else default_r

    for k in range(4):
        pos_pts = positive_coords_k.get(k)
        if pos_pts is None or len(pos_pts) == 0:
            continue

        pos_pts = np.asarray(pos_pts, dtype=np.float64)
        if pos_pts.ndim == 1:
            pos_pts = pos_pts.reshape(1, -1)

        if is_geographic:
            pos_metric = np.column_stack([pos_pts[:, 0] * m_per_deg_lon, pos_pts[:, 1] * m_per_deg_lat])
        else:
            pos_metric = pos_pts

        if k == 3 and isinstance(radii[3], (tuple, list)):
            # Anisotropic ellipse: 10 km along-strike x 2 km across-strike, strike 45° (NE-SW)
            a, b, strike_deg = radii[3]
            theta_rad = np.radians(strike_deg)
            sin_t = np.sin(theta_rad)
            cos_t = np.cos(theta_rad)

            # Query cells within bounding radius a around any P3 point
            candidate_indices = tree.query_ball_point(pos_metric, r=float(a))
            for p_idx, cell_indices in enumerate(candidate_indices):
                if not cell_indices:
                    continue
                px_m, py_m = pos_metric[p_idx]
                dx = metric_xy[cell_indices, 0] - px_m
                dy = metric_xy[cell_indices, 1] - py_m
                u = dx * sin_t + dy * cos_t   # along-strike
                v = dx * cos_t - dy * sin_t   # across-strike
                in_ellipse = (u / a) ** 2 + (v / b) ** 2 <= 1.0
                masked_cells = np.array(cell_indices)[in_ellipse]
                if len(masked_cells) > 0:
                    if labels_k4 is not None:
                        # Only mask unlabeled cells (label == -1)
                        is_unl = (labels_k4[masked_cells, k] == -1)
                        M[masked_cells[is_unl], k] = 0.0
                    else:
                        M[masked_cells, k] = 0.0
        else:
            # Isotropic circular buffer
            r_val = float(radii[k])
            matched_indices = tree.query_ball_point(pos_metric, r=r_val)
            flat_indices = np.unique([idx for sublist in matched_indices for idx in sublist])
            if len(flat_indices) > 0:
                if labels_k4 is not None:
                    is_unl = (labels_k4[flat_indices, k] == -1)
                    M[flat_indices[is_unl], k] = 0.0
                else:
                    M[flat_indices, k] = 0.0

    return M


def compute_asymmetric_head_weights(
    labels_k: np.ndarray,
    e_logistics: np.ndarray,
    exclusion_mask_k: np.ndarray,
    min_propensity: float = 0.05,
) -> np.ndarray:
    """Compute asymmetric 1D sample weights w_k for Head k.

    Enforces Task 1.3 Rule 3 and Reconciled Engineering Rule 5:
      - Positive cells (label_k == 1): receive inverse propensity weighting:
            w_k[pos_mask] = 1.0 / np.clip(e_logistics[pos_mask], 0.05, 1.0)
      - Unlabeled cells (label_k == -1): receive r_k* halo mask * exploration-confidence weight:
            w_k[unl_mask] = M[unl_mask, k] * (0.25 + 0.75 * e_logistics[unl_mask])

    Eliminates the double-division trap where unexplored jungle is penalised 20x relative
    to roads.

    Args:
        labels_k: 1-D array of shape (N,) with labels in {1, -1}.
        e_logistics: 1-D array of shape (N,) of observation propensities e(s).
        exclusion_mask_k: 1-D array of shape (N,) representing column k of M in {0.0, 1.0}.
        min_propensity: Floor on observation propensity for positive weighting (default 0.05).

    Returns:
        1-D float64 array w_k of shape (N,).
    """
    labels = np.asarray(labels_k).ravel()
    e_prop = np.asarray(e_logistics, dtype=np.float64).ravel()
    m_mask = np.asarray(exclusion_mask_k, dtype=np.float64).ravel()

    if len(labels) != len(e_prop) or len(labels) != len(m_mask):
        raise ValueError(
            f"Dimension mismatch: labels ({len(labels)}), e_logistics ({len(e_prop)}), exclusion_mask ({len(m_mask)})"
        )

    w_k = np.zeros(len(labels), dtype=np.float64)
    pos_mask = (labels == 1)
    unl_mask = (labels == -1)

    # Positive cells (label_k == 1): inverse propensity weighting
    w_k[pos_mask] = 1.0 / np.clip(e_prop[pos_mask], float(min_propensity), 1.0)

    # Unlabeled cells (label_k == -1): r_k* halo masked + exploration-confidence weighted
    w_k[unl_mask] = m_mask[unl_mask] * (0.25 + 0.75 * e_prop[unl_mask])

    return w_k


class OUTMOGCGate:
    """Outlier Mitigation & Gated Calibration (OUTMOGC) closed-boundary gate.

    Enforces Task 1.3 Rule 4:
      Fits a 4-component PCA(n_components=4, whiten=True) + 2-component
      GaussianMixture(n_components=2, covariance_type="full") on the Merdith
      geophysical features of each positive set P_k to compute Gamma_closed^(k)(s) in (0, 1]
      via exponential Mahalanobis distance decay above chi^2_{4, 0.95} (~ 9.488).

    For points inside the 95% confidence ellipsoid (d^2 <= chi^2_{4, 0.95}):
      Gamma_closed = 1.0
    For points outside (d^2 > chi^2_{4, 0.95}):
      Gamma_closed = exp(-0.5 * (d^2 - chi^2_{4, 0.95}))
    """

    def __init__(
        self,
        n_pca_components: int = 4,
        n_gmm_components: int = 2,
        random_state: int = 42,
    ) -> None:
        self.n_pca_components = n_pca_components
        self.n_gmm_components = n_gmm_components
        self.random_state = random_state
        self.pca = None
        self.gmm = None
        self.chi2_crit = None
        self.actual_df = None

    def fit(self, X_positives: np.ndarray) -> OUTMOGCGate:
        """Fit PCA and GMM on the Merdith geophysical features of positive set P_k.

        Args:
            X_positives: 2-D array of shape (N_p, n_features).

        Returns:
            Fitted self.
        """
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        from scipy.stats import chi2

        X = np.asarray(X_positives, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X_positives must be 2-D, got shape {X.shape}")

        n_samples, n_features = X.shape
        actual_pca_comp = max(1, min(self.n_pca_components, n_samples - 1, n_features))
        self.actual_df = actual_pca_comp
        self.chi2_crit = float(chi2.ppf(0.95, df=self.actual_df))

        self.pca = PCA(n_components=actual_pca_comp, whiten=True, random_state=self.random_state)
        Z = self.pca.fit_transform(X)

        actual_gmm_comp = max(1, min(self.n_gmm_components, n_samples // 2))
        self.gmm = GaussianMixture(
            n_components=actual_gmm_comp,
            covariance_type="full",
            random_state=self.random_state,
        )
        self.gmm.fit(Z)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Gamma_closed^(k)(s) in (0, 1] on query cells.

        Args:
            X: 2-D feature array of shape (N, n_features).

        Returns:
            1-D float64 array of shape (N,) with Gamma_closed in (0, 1].
        """
        if self.pca is None or self.gmm is None:
            raise RuntimeError("OUTMOGCGate is not fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        Z = self.pca.transform(X_arr)

        d2_components = []
        for c in range(self.gmm.n_components):
            diff = Z - self.gmm.means_[c]
            cov = self.gmm.covariances_[c]
            # Ridge regularization for singular-free matrix inversion
            cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
            inv_cov = np.linalg.pinv(cov_reg)
            d2_c = np.sum(diff @ inv_cov * diff, axis=1)
            d2_components.append(d2_c)

        d2_min = np.min(d2_components, axis=0)
        # Inside gate (d^2 <= chi2_crit): Gamma = 1.0; outside: exponential decay
        gamma = np.where(
            d2_min <= self.chi2_crit,
            1.0,
            np.exp(-0.5 * (d2_min - self.chi2_crit))
        )
        return np.clip(gamma, 1e-12, 1.0).astype(np.float64)


