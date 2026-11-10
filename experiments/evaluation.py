"""Alert evaluation, identical for both methods, plus bootstrap confidence intervals.

Each method produces a boolean mask (run, day, facility).
- Onset: a facility enters the alert set. Onsets closer than the refractory period form
  one alert event. An alert already active before an outbreak earns no credit for it.
- Detection: the first onset inside [t0, t0 + duration + grace) at an outbreak facility.
- True event: an event containing such an onset. Every other event is a false alert.
"""
import numpy as np
import pandas as pd

METRIC_COLUMNS = ["detection_rate", "recall", "detection_delay_days", "restricted_delay_days",
                  "false_alerts_per_year", "precision", "alert_burden_null",
                  "messages_per_day", "chance_detection", "attributable_detection_rate",
                  "attributable_restricted_delay_days", "mean_alert_size"]


def evaluate_mask(mask, burn_in, refractory, t0=None, t_end=None, cluster=None):
    """Return per-run arrays. Omit the outbreak arguments for null runs."""
    R, T, _ = mask.shape
    onset = mask.copy()
    onset[:, 1:] &= ~mask[:, :-1]
    onset[:, :burn_in] = False
    any_onset = onset.any(axis=2)
    n_events = np.zeros(R, int)
    n_false = np.zeros(R, int)
    detection_day = np.full(R, -1, int)
    for r in np.nonzero(any_onset.any(axis=1))[0]:
        days = np.nonzero(any_onset[r])[0]
        groups = np.split(days, np.nonzero(np.diff(days) > refractory)[0] + 1)
        n_events[r] = len(groups)
        true_days = np.array([], int)
        if cluster is not None:
            window = np.zeros(T, bool)
            window[t0[r]:t_end[r]] = True
            true_days = np.nonzero(window & onset[r][:, cluster[r]].any(axis=1))[0]
            if len(true_days):
                detection_day[r] = true_days[0]
        n_false[r] = sum(not np.isin(g, true_days).any() for g in groups)
    return {"n_events": n_events, "n_false": n_false, "detection_day": detection_day,
            "burden": mask[:, burn_in:].any(axis=2).mean(axis=1)}


def attributable_detection_day(mask_out, mask_null, t0, t_end, cluster, tolerance):
    """First onset at an outbreak site in the window with no onset at an outbreak site within
    `tolerance` days on the outbreak-free twin run. Because the twins share all noise, such an
    onset is attributable to the outbreak and not to coincidence. -1 if none exists."""
    def onsets(mask):
        out = mask.copy()
        out[:, 1:] &= ~mask[:, :-1]
        return out

    on_out, on_null = onsets(mask_out), onsets(mask_null)
    day = np.full(len(mask_out), -1, int)
    for r in range(len(mask_out)):
        days = np.nonzero(on_out[r, t0[r]:t_end[r]][:, cluster[r]].any(axis=1))[0] + t0[r]
        if not len(days):
            continue
        twin = np.nonzero(on_null[r][:, cluster[r]].any(axis=1))[0]
        for d in days:
            if not len(twin) or np.abs(twin - d).min() > tolerance:
                day[r] = d
                break
    return day


def _ci(samples):
    return np.nanpercentile(samples, [2.5, 97.5])


def _ratio_samples(numerator, denominator, idx):
    n, d = numerator[idx].sum(axis=1), denominator[idx].sum(axis=1)
    return np.where(d > 0, n / np.maximum(d, 1), np.nan)


def bootstrap_metrics(df, n_boot, seed):
    """Percentile bootstrap over seeds (one row per seed). Precision and conditional delay
    are recomputed from resampled totals, so they stay well defined when a seed has no alert."""
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = rng.integers(0, n, (n_boot, n))
    rows = []
    for metric in METRIC_COLUMNS:
        if metric == "precision":
            tp, fp = df["tp"].to_numpy(float), df["fp"].to_numpy(float)
            point = tp.sum() / (tp.sum() + fp.sum()) if tp.sum() + fp.sum() else np.nan
            samples = _ratio_samples(tp, tp + fp, idx)
            n_used = int(tp.sum() + fp.sum())
        elif metric in ("detection_delay_days", "mean_alert_size"):
            delay = df[metric].to_numpy(float)
            ok = ~np.isnan(delay)
            point = delay[ok].mean() if ok.any() else np.nan
            samples = _ratio_samples(np.nan_to_num(delay), ok.astype(float), idx)
            n_used = int(ok.sum())
        else:
            values = df[metric].to_numpy(float)
            point = values.mean()
            samples = values[idx].mean(axis=1)
            n_used = n
        low, high = _ci(samples) if np.isfinite(samples).any() else (np.nan, np.nan)
        rows.append({"metric": metric, "mean": point, "ci_low": low, "ci_high": high, "n": n_used})
    return pd.DataFrame(rows)


def paired_difference(a, b, metric, n_boot, seed):
    """Bootstrap CI for mean(a - b) over seeds. Both frames must share seed order."""
    rng = np.random.default_rng(seed)
    a_values, b_values = a[metric].to_numpy(float), b[metric].to_numpy(float)
    diff = a_values - b_values
    idx = rng.integers(0, len(diff), (n_boot, len(diff)))
    low, high = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return {"difference": diff.mean(), "ci_low": low, "ci_high": high,
            "ci_excludes_zero": bool(low > 0 or high < 0)}
