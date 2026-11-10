"""Pooled Isolation Forest and calibration of scores into signal production."""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest


def fit_forest(X_train, cfg):
    """Fit on normal data only. `contamination` is unused: we never threshold here."""
    model = IsolationForest(n_estimators=cfg.n_estimators, max_samples=cfg.max_samples,
                            max_features=cfg.max_features, random_state=cfg.random_state,
                            n_jobs=-1)
    return model.fit(X_train)


def raw_scores(model, X):
    """Higher means more anomalous."""
    return -model.score_samples(X)


@dataclass
class ScoreCalibrator:
    """Maps a score to P = max(0, ln(tail / p)), where p is the upper-tail probability of the
    score under that facility's own normal calibration data. On normal data P is zero for the
    bottom (1 - tail) share of a site's reports and has mean about `tail`, for any site and any
    score. So the forest and the non-ML score get the same null production budget, and a large
    facility gets no more null signal than a small one. Each site can calibrate on its own."""
    sorted_scores: list
    n_total: np.ndarray
    tail: float

    @classmethod
    def fit(cls, scores, excess, sites, n_sites, tail):
        by_site = [np.sort(scores[(sites == k) & (excess > 0)]) for k in range(n_sites)]
        return cls(by_site, np.bincount(sites, minlength=n_sites), tail)

    def tail_probability(self, scores, sites):
        p = np.ones(len(scores))
        for k, reference in enumerate(self.sorted_scores):
            here = sites == k
            n_above = len(reference) - np.searchsorted(reference, scores[here], "left")
            p[here] = (1 + n_above) / (self.n_total[k] + 1)
        return p

    def production(self, scores, excess, sites):
        p = self.tail_probability(scores, sites)
        return np.where(excess > 0, np.maximum(0.0, np.log(self.tail / p)), 0.0)
