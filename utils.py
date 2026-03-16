from pathlib import Path

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
# CONFIG — edit these
# ─────────────────────────────────────────────
HORIZONS = [24, 48]
HORIZON_WEIGHTS = [0.3, 0.4]
TARGETS = ['time_to_hit_hours', 'event']
N_SPLITS = 5
RANDOM_STATE = 42
PENALIZER = 0.1         # L2 regularisation for Cox
MAX_OBSERVED_TIME = 67   # max time in training data

# Calibration method: 'platt' or 'isotonic'
CALIBRATION_METHOD = 'platt'


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_survival_probs(cph, df, horizons, max_observed):
    """
    Predict survival probabilities at each horizon.
    Clips horizon to max_observed to avoid Breslow extrapolation,
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

def get_log_hazard(cph, df, features):
    """
    Get log hazard predictions from Cox model.
    Used for C-index calculation (relative risk ranking).
    """
    return cph.predict_log_partial_hazard(df[features]).values

def get_horizon_labels(df, horizon):
    """
    Returns mask and binary labels for calibration at a given horizon.

    Fires censored BEFORE the horizon are excluded — their outcome is
    unknown (we can't tell if they would have hit by the horizon).

    Only two kinds of rows have a known outcome:
      - event=1  AND time <= horizon  → definitely hit by horizon  (label=1)
      - event=1  AND time >  horizon  → definitely missed horizon  (label=0)
      - event=0  AND time >  horizon  → survived past horizon      (label=0)
      - event=0  AND time <= horizon  → CENSORED before horizon    → EXCLUDED

    Parameters
    ----------
    df      : pd.DataFrame with 'event' and 'time_to_hit_hours' columns
    horizon : int, horizon in hours

    Returns
    -------
    mask   : boolean array, True = row has known outcome at this horizon
    labels : int array, 1 if hit by horizon else 0 (only valid where mask=True)
    """
    mask   = (df['event'] == 1) | (df['time_to_hit_hours'] > horizon)
    labels = ((df['event'] == 1) & (df['time_to_hit_hours'] <= horizon)).astype(int)
    return mask.values, labels.values


def fit_calibrators(raw_probs_dict, df, method='platt'):
    """
    Fit one calibrator per horizon on OOF predictions.
    Rows censored before the horizon are excluded from calibration fitting.

    Parameters
    ----------
    raw_probs_dict : dict {horizon: np.array of raw probabilities}
    df             : pd.DataFrame with 'event' and 'time_to_hit_hours' columns
                     (must be the full OOF dataframe, same row order as raw_probs_dict)
    method         : 'platt' or 'isotonic'

    Returns
    -------
    calibrators : dict {horizon: fitted calibrator}
    """
    calibrators = {}
    for h, probs in raw_probs_dict.items():
        mask, labels = get_horizon_labels(df, h)

        # skip calibration if only one class present after masking
        if len(np.unique(labels[mask])) < 2:
            print(f"  Calibrator @{h}h: skipped — only one class after censoring mask")
            calibrators[h] = None
            continue

        X_cal = probs[mask].reshape(-1, 1)
        y_cal = labels[mask]

        if method == 'platt':
            cal = LogisticRegression(C=1.0, random_state=42)
            cal.fit(X_cal, y_cal)
        elif method == 'isotonic':
            cal = IsotonicRegression(out_of_bounds='clip')
            cal.fit(X_cal.flatten(), y_cal)

        calibrators[h] = cal
        print(f"  Calibrator @{h}h: {mask.sum()} usable rows | "
              f"{y_cal.sum()} hits | base rate {y_cal.mean():.3f}")

    return calibrators


def apply_calibrators(calibrators, raw_probs_dict, method='platt'):
    """
    Apply fitted calibrators to raw probabilities.
    Applied to ALL rows (no masking needed at prediction time).

    Returns
    -------
    dict {horizon: calibrated np.array}
    """
    calibrated = {}
    for h, cal in calibrators.items():
        probs = raw_probs_dict[h].reshape(-1, 1)
        if cal is None:
            calibrated[h] = probs.flatten()
            continue
        elif method == 'platt':
            calibrated[h] = cal.predict_proba(probs)[:, 1]
        elif method == 'isotonic':
            calibrated[h] = cal.predict(probs.flatten())
    return calibrated

def fit_calibrators_hazard(oof_hazard, df, method='platt'):
    """
    Fit one calibrator per horizon on OOF log partial hazard scores.
    Rows censored before the horizon are excluded from calibration fitting.

    Parameters
    ----------
    oof_hazard : np.array of log partial hazard scores (1D, one per row)
    df         : pd.DataFrame with 'event' and 'time_to_hit_hours' columns
    method     : 'platt' or 'isotonic'

    Returns
    -------
    calibrators : dict {horizon: fitted calibrator or None}
    """
    calibrators = {}
    for h in HORIZONS:
        mask, labels = get_horizon_labels(df, h)

        # skip if only one class present after masking
        if len(np.unique(labels[mask])) < 2:
            print(f"  Calibrator @{h}h: skipped — only one class after censoring mask")
            calibrators[h] = None
            continue

        X_cal = oof_hazard[mask].reshape(-1, 1)
        y_cal = labels[mask]

        if method == 'platt':
            cal = LogisticRegression(C=1.0, random_state=42)
            cal.fit(X_cal, y_cal)
        elif method == 'isotonic':
            cal = IsotonicRegression(out_of_bounds='clip')
            cal.fit(X_cal.flatten(), y_cal)

        calibrators[h] = cal
        print(f"  Calibrator @{h}h: {mask.sum()} usable rows | "
              f"{y_cal.sum()} hits | base rate {y_cal.mean():.3f}")

    return calibrators


def apply_calibrators_hazard(calibrators, hazard_scores, method='platt'):
    """
    Apply fitted calibrators to log partial hazard scores.
    Returns a dict of calibrated probabilities per horizon.

    Parameters
    ----------
    calibrators   : dict {horizon: fitted calibrator or None}
    hazard_scores : np.array of log partial hazard scores (1D)
    method        : 'platt' or 'isotonic'

    Returns
    -------
    dict {horizon: calibrated np.array of probabilities}
    """
    calibrated = {}
    scores = hazard_scores.reshape(-1, 1)
    for h, cal in calibrators.items():
        if cal is None:
            # fallback — return uniform base rate, or just zeros
            calibrated[h] = np.full(len(hazard_scores), np.nan)
            continue
        if method == 'platt':
            calibrated[h] = cal.predict_proba(scores)[:, 1]
        elif method == 'isotonic':
            calibrated[h] = cal.predict(scores.flatten())
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
    if verbose:
        print("=" * 80)
        print(f"CV: {feature_name}  |  {len(feature_list)} features  |  {N_SPLITS}-fold stratified")
        print("=" * 80)

    data_subset = data[TARGETS + feature_list].copy().reset_index(drop=True)
    events = data_subset['event'].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_scores = []

    # Storage for OOF log hazards (for calibration) and raw probs (for Brier)
    oof_hazard = np.zeros(len(data_subset))
    oof_raw = {h: np.zeros(len(data_subset)) for h in HORIZONS}

    # ── Pass 1: collect OOF log hazards and raw probs ──
    for fold, (train_idx, val_idx) in enumerate(skf.split(data_subset, events)):
        train_fold = data_subset.iloc[train_idx].copy()
        val_fold   = data_subset.iloc[val_idx].copy()

        cph = CoxPHFitter(penalizer=PENALIZER)
        cph.fit(train_fold, duration_col='time_to_hit_hours', event_col='event')

        oof_hazard[val_idx] = get_log_hazard(cph, val_fold, feature_list)

        raw_probs = get_survival_probs(cph, val_fold[feature_list], HORIZONS, MAX_OBSERVED_TIME)
        for h in HORIZONS:
            oof_raw[h][val_idx] = raw_probs[h]

    # ── Fit calibrators on OOF log hazards (censoring-aware per horizon) ──
    print("\nFitting calibrators on OOF predictions:")
    calibrators = fit_calibrators_hazard(oof_hazard, data_subset, method=CALIBRATION_METHOD)

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

        # Calibrate using log hazard
        val_hazard = get_log_hazard(cph, val_fold, feature_list)
        cal_probs = apply_calibrators_hazard(calibrators, val_hazard, method=CALIBRATION_METHOD)

        # Brier score using survival probs
        raw_probs = get_survival_probs(cph, val_fold[feature_list], HORIZONS, MAX_OBSERVED_TIME)

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

    if verbose:
        print("\n" + "=" * 80)
        print(f"SUMMARY — {feature_name}")
        print("=" * 80)
        for metric in ['c_index', 'brier_24h', 'brier_48h', 'weighted_brier', 'hybrid']:
            vals = [f[metric] for f in fold_scores if metric in f]
            if vals:
                print(f"  {metric:<22}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return fold_scores, None


# ─────────────────────────────────────────────
# TRAIN + SUBMIT
# ─────────────────────────────────────────────

def train_and_submit(feature_list, data, test_data, filename_suffix=""):
    """
    Train on full data with OOF-fitted calibrators and generate submission.
    Calibrators are fit on OOF log partial hazard scores (no leakage).
    """
    print("=" * 80)
    print(f"TRAINING ON FULL DATA — {len(feature_list)} features")
    print("=" * 80)

    data_subset = data[TARGETS + feature_list].copy().reset_index(drop=True)
    events = data_subset['event'].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Collect OOF log hazards for calibration fitting
    oof_hazard = np.zeros(len(data_subset))

    for train_idx, val_idx in skf.split(data_subset, events):
        train_fold = data_subset.iloc[train_idx].copy()
        val_fold   = data_subset.iloc[val_idx].copy()

        cph = CoxPHFitter(penalizer=PENALIZER)
        cph.fit(train_fold, duration_col='time_to_hit_hours', event_col='event')

        oof_hazard[val_idx] = get_log_hazard(cph, val_fold, feature_list)

    print("\nFitting calibrators on OOF predictions:")
    calibrators = fit_calibrators_hazard(oof_hazard, data_subset, method=CALIBRATION_METHOD)

    # Final model on full data
    cph_final = CoxPHFitter(penalizer=PENALIZER)
    cph_final.fit(data_subset, duration_col='time_to_hit_hours', event_col='event')

    # Predict on test
    test_features = test_data[feature_list].copy()
    test_hazard = get_log_hazard(cph_final, test_data, feature_list)
    cal_test = apply_calibrators_hazard(calibrators, test_hazard, method=CALIBRATION_METHOD)

    # Build submission
    predictions = {'event_id': test_data['event_id']}

    # 12h: raw Cox, no calibrator
    raw_12h = get_survival_probs(cph_final, test_features, [12], MAX_OBSERVED_TIME)
    predictions['prob_12h'] = raw_12h[12]

    for h in [24, 48]:
        predictions[f'prob_{h}h'] = cal_test[h]
        print(f"prob_{h}h — mean={cal_test[h].mean():.3f}  min={cal_test[h].min():.3f}  max={cal_test[h].max():.3f}")

    # 72h: use raw Cox probability at 67h as a proxy (no calibrator, no extrapolation)
    raw_72h = get_survival_probs(cph_final, test_features, [72], MAX_OBSERVED_TIME)
    predictions['prob_72h'] = raw_72h[72]
    print(f"prob_72h (proxy) — mean={cal_test[48].mean():.3f}  min={cal_test[48].min():.3f}  max={cal_test[48].max():.3f}")

    submission = pd.DataFrame(predictions)
    filename = f'test_submission{filename_suffix}.csv'
    submission.to_csv(Path("submissions") / filename, index=False)
    print(f"\nSaved: {filename}  |  shape: {submission.shape}")
    print(submission.head())

    return cph_final, calibrators, submission


# ─────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────

# features = ['cross_track_component', 'log_area_epsilonp']
# fold_scores, oof_probs = evaluate_features(features, full_train_data, feature_name="2-feature baseline")
# cph, calibrators, submission = train_and_submit(features, full_train_data, test, filename_suffix="_v1")