"""Isolation Forest with no bacterial layer.

Scores use the same trained forest as the full model. Each is standardised against that site's
own normal calibration scores, and only excess (not deficit) reports can score. A centre pools its
closed neighbourhood's scores that day, as sum / sqrt(n reporting), and the whole neighbourhood
alerts when this reaches the threshold. There is no signal decay, accumulation, Hill activation,
hysteresis or quorum vote. The forest's own rolling features still carry some memory.
"""
import numpy as np

from data.reporting import count_by_arrival, max_by_arrival


def site_scores(score_ml, excess, arrival, site_mean, site_sd):
    """Daily standardised score per site, shape (R, T, N). Absent or deficit reports are -inf."""
    runs = []
    for raw, up, arr in zip(score_ml, excess, arrival):
        stat = np.where((up > 0) & ~np.isnan(raw), (raw.astype(float) - site_mean) / site_sd, -np.inf)
        runs.append(max_by_arrival(stat, arr))
    return np.stack(runs).astype(np.float32)


def neighbourhood_scores(scores, arrival, neighbourhoods):
    """Pooled score per centre, shape (R, T, N). Silent or deficit sites contribute zero."""
    Nb = neighbourhoods.astype(np.float32)
    clean = np.where(np.isfinite(scores), np.maximum(scores, 0.0), 0.0).astype(np.float32)
    reporting = np.stack([(count_by_arrival(a) > 0).astype(np.float32) for a in arrival]) @ Nb
    return (clean @ Nb) / np.sqrt(np.maximum(reporting, 1.0))


def neighbourhood_mask(pooled, threshold, neighbourhoods):
    """A firing centre puts its whole closed neighbourhood into the alert set."""
    return ((pooled >= threshold).astype(np.float32) @ neighbourhoods.astype(np.float32)) > 0
