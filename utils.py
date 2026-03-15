import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.metrics import brier_score
from sksurv.util import Surv
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────
HORIZONS = [24, 48, 72]
HORIZON_WEIGHTS = [0.3, 0.4, 0.3]
TARGETS = ['time_to_hit_hours', 'event']
N_SPLITS = 5
RANDOM_STATE = 42
PENALIZER = 0.1          # L2 regularisation for Cox
MAX_OBSERVED_TIME = 67   # max time in training data

# Calibration method: 'platt' or 'isotonic'
CALIBRATION_METHOD = 'platt'


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_survival_probs(cph, df, horizons, max_observed):
    """
    Predict survival probabilities at each horizon.
    Clips horizon to max_observed to avoid extrapolation,
    then returns P(event by horizon) = 1 - S(horizon).
    """
    probs = {}
    for h in horizons:
        # Clip to max observed time to avoid extrapolation instability.
        # For 72h this will use S(67) as the best available estimate.
        safe_h = min(h, max_observed)
        sf = cph.predict_survival_function(df, times=[safe_h]).values.flatten()
        probs[h] = 1 - sf
    return probs  # dict: {24: array, 48: array, 72: array}


def fit_calibrators(raw_probs_dict, true_events, method='platt'):
    """
    Fit one calibrator per horizon on OOF predictions.

    Parameters
    ----------
    raw_probs_dict : dict {horizon: np.array of raw probabilities}
    true_events    : np.array of binary event labels (1=hit, 0=censored/no hit)
    method         : 'platt' or 'isotonic'

    Returns
    -------
    calibrators : dict {horizon: fitted calibrator}
    """
    calibrators = {}
    for h, probs in raw_probs_dict.items():
        probs = probs.reshape(-1, 1)
        if method == 'platt':
            cal = LogisticRegression(C=1e10)  # effectively just sigmoid fitting
            cal.fit(probs, true_events)
        elif method == 'isotonic':
            cal = IsotonicRegression(out_of_bounds='clip')
            cal.fit(probs.flatten(), true_events)
        calibrators[h] = cal
    return calibrators


def apply_calibrators(calibrators, raw_probs_dict, method='platt'):
    """
    Apply fitted calibrators to raw probabilities.

    Returns
    -------
    dict {horizon: calibrated np.array}
    """
    calibrated = {}
    for h, cal in calibrators.items():
        probs = raw_probs_dict[h].reshape(-1, 1)
        if method == 'platt':
            calibrated[h] = cal.predict_proba(probs)[:, 1]
        elif method == 'isotonic':
            calibrated[h] = cal.predict(probs.flatten())
    return calibrated


def compute_brier_scores(calibrated_probs, train_surv, val_surv, val_times, horizons, weights, max_observed):
    """
    Compute censoring-aware Brier score at each horizon using sksurv.
    Only evaluates at horizons that are within the observed time range of the val fold.
    Returns per-horizon scores and the weighted average.
    """
    brier_results = {}
    for h, w in zip(horizons, weights):
        safe_h = min(h, max_observed)

        # Skip if safe_h exceeds the val fold's max observed time
        # (sksurv will error if the time point is beyond the IPCW support)
        val_max = val_times.max()
        if safe_h > val_max:
            # Use val_max as fallback — still evaluates calibration near the horizon
            safe_h = val_max * 0.999  # just inside the support

        probs = calibrated_probs[h]
        _, bs = brier_score(train_surv, val_surv, probs, [safe_h])
        brier_results[h] = (w, bs[0])

    total_weight = sum(w for w, _ in brier_results.values())
    weighted_brier = sum(w * bs for w, bs in brier_results.values()) / total_weight
    return brier_results, weighted_brier


def compute_hybrid(c_index, weighted_brier):
    return 0.3 * c_index + 0.7 * (1 - weighted_brier)


# ─────────────────────────────────────────────
# MAIN CV FUNCTION
# ─────────────────────────────────────────────

def evaluate_features(feature_list, data, feature_name="Model", verbose=True):
    """
    Full cross-validation pipeline with:
    - StratifiedKFold on event column
    - Cox PH with penalisation
    - Per-horizon Platt/isotonic calibration fit on OOF predictions
    - Censoring-aware Brier score via sksurv
    - Exact competition hybrid metric

    Parameters
    ----------
    feature_list : list of str
    data         : pd.DataFrame with TARGETS + features
    feature_name : str for printing
    verbose      : bool

    Returns
    -------
    fold_scores : list of dicts with per-fold metrics
    oof_probs   : dict {horizon: np.array} of OOF calibrated probabilities
                  (useful for stacking or further analysis)
    """
    if verbose:
        print("=" * 80)
        print(f"CV: {feature_name}  |  {len(feature_list)} features  |  {N_SPLITS}-fold stratified")
        print("=" * 80)

    data_subset = data[TARGETS + feature_list].copy().reset_index(drop=True)
    events = data_subset['event'].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_scores = []

    # Storage for OOF raw probabilities (before calibration)
    oof_raw = {h: np.zeros(len(data_subset)) for h in HORIZONS}
    oof_events = np.zeros(len(data_subset))

    # ── Pass 1: collect OOF raw probabilities ──
    for fold, (train_idx, val_idx) in enumerate(skf.split(data_subset, events)):
        train_fold = data_subset.iloc[train_idx].copy()
        val_fold   = data_subset.iloc[val_idx].copy()

        cph = CoxPHFitter(penalizer=PENALIZER)
        cph.fit(train_fold, duration_col='time_to_hit_hours', event_col='event')

        raw_probs = get_survival_probs(cph, val_fold[feature_list], HORIZONS, MAX_OBSERVED_TIME)

        for h in HORIZONS:
            oof_raw[h][val_idx] = raw_probs[h]
        oof_events[val_idx] = val_fold['event'].values

    # ── Fit calibrators on full OOF predictions ──
    calibrators = fit_calibrators(oof_raw, oof_events, method=CALIBRATION_METHOD)

    # ── Pass 2: evaluate with calibrated probabilities ──
    for fold, (train_idx, val_idx) in enumerate(skf.split(data_subset, events)):
        train_fold = data_subset.iloc[train_idx].copy()
        val_fold   = data_subset.iloc[val_idx].copy()

        cph = CoxPHFitter(penalizer=PENALIZER)
        cph.fit(train_fold, duration_col='time_to_hit_hours', event_col='event')

        # C-index
        c_idx = concordance_index(
            val_fold['time_to_hit_hours'],
            -cph.predict_partial_hazard(val_fold[feature_list]),
            val_fold['event']
        )

        # Raw probs → calibrate
        raw_probs = get_survival_probs(cph, val_fold[feature_list], HORIZONS, MAX_OBSERVED_TIME)
        cal_probs = apply_calibrators(calibrators, raw_probs, method=CALIBRATION_METHOD)

        # Brier score
        train_surv = Surv.from_arrays(
            event=train_fold['event'].astype(bool),
            time=train_fold['time_to_hit_hours']
        )
        val_surv = Surv.from_arrays(
            event=val_fold['event'].astype(bool),
            time=val_fold['time_to_hit_hours']
        )

        brier_results, weighted_brier = compute_brier_scores(
            cal_probs, train_surv, val_surv,
            val_fold['time_to_hit_hours'], HORIZONS, HORIZON_WEIGHTS, MAX_OBSERVED_TIME
        )

        hybrid = compute_hybrid(c_idx, weighted_brier)

        fold_result = {
            'fold': fold + 1,
            'c_index': c_idx,
            'weighted_brier': weighted_brier,
            'hybrid': hybrid,
            **{f'brier_{h}h': bs for h, (_, bs) in brier_results.items()}
        }
        fold_scores.append(fold_result)

        if verbose:
            brier_str = " | ".join(f"Brier@{h}h={bs:.4f}" for h, (_, bs) in brier_results.items())
            print(f"Fold {fold+1}: C-index={c_idx:.4f} | {brier_str} | WBrier={weighted_brier:.4f} | Hybrid={hybrid:.4f}")

    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print(f"SUMMARY — {feature_name}")
        print("=" * 80)
        for metric in ['c_index', 'brier_24h', 'brier_48h', 'brier_72h', 'weighted_brier', 'hybrid']:
            vals = [f[metric] for f in fold_scores if metric in f]
            if vals:
                print(f"  {metric:<22}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    oof_calibrated = apply_calibrators(calibrators, oof_raw, method=CALIBRATION_METHOD)
    return fold_scores, oof_calibrated


# ─────────────────────────────────────────────
# TRAIN + SUBMIT
# ─────────────────────────────────────────────

def train_and_submit(feature_list, data, test_data, filename_suffix=""):
    """
    Train on full data with OOF-fitted calibrators and generate submission.
    Calibrators are fit on OOF predictions from the full training set (no leakage).
    """
    print("=" * 80)
    print(f"TRAINING ON FULL DATA — {len(feature_list)} features")
    print("=" * 80)

    data_subset = data[TARGETS + feature_list].copy().reset_index(drop=True)
    events = data_subset['event'].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Collect OOF raw probs for calibration fitting
    oof_raw = {h: np.zeros(len(data_subset)) for h in HORIZONS}
    oof_events = np.zeros(len(data_subset))

    for train_idx, val_idx in skf.split(data_subset, events):
        train_fold = data_subset.iloc[train_idx].copy()
        val_fold   = data_subset.iloc[val_idx].copy()

        cph = CoxPHFitter(penalizer=PENALIZER)
        cph.fit(train_fold, duration_col='time_to_hit_hours', event_col='event')

        raw_probs = get_survival_probs(cph, val_fold[feature_list], HORIZONS, MAX_OBSERVED_TIME)
        for h in HORIZONS:
            oof_raw[h][val_idx] = raw_probs[h]
        oof_events[val_idx] = val_fold['event'].values

    calibrators = fit_calibrators(oof_raw, oof_events, method=CALIBRATION_METHOD)

    # Final model on full data
    cph_final = CoxPHFitter(penalizer=PENALIZER)
    cph_final.fit(data_subset, duration_col='time_to_hit_hours', event_col='event')

    # Predict on test
    test_features = test_data[feature_list].copy()
    raw_test = get_survival_probs(cph_final, test_features, HORIZONS, MAX_OBSERVED_TIME)
    cal_test = apply_calibrators(calibrators, raw_test, method=CALIBRATION_METHOD)

    # Build submission
    submission_horizons = [12, 24, 48, 72]
    predictions = {'event_id': test_data['event_id']}

    # 12h: not in calibration horizons, use raw Cox directly (no calibrator for this)
    raw_12h = get_survival_probs(cph_final, test_features, [12], MAX_OBSERVED_TIME)
    predictions['prob_12h'] = raw_12h[12]

    for h in [24, 48, 72]:
        predictions[f'prob_{h}h'] = cal_test[h]
        print(f"prob_{h}h — mean={cal_test[h].mean():.3f}  min={cal_test[h].min():.3f}  max={cal_test[h].max():.3f}")

    submission = pd.DataFrame(predictions)
    filename = f'test_submission{filename_suffix}.csv'
    submission.to_csv(filename, index=False)
    print(f"\nSaved: {filename}  |  shape: {submission.shape}")
    print(submission.head())

    return cph_final, calibrators, submission


# ─────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────

# features = ['cross_track_component', 'log_area_epsilonp']
# fold_scores, oof_probs = evaluate_features(features, full_train_data, feature_name="2-feature baseline")
# cph, calibrators, submission = train_and_submit(features, full_train_data, test, filename_suffix="_v1")