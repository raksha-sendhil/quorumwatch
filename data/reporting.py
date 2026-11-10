"""Reporting process: missing reports, delayed reports and data-entry artefacts.

`arrival[s, i]` is the day on which the report for day s from facility i reaches
the system, or -1 if it never does. Detectors may only use a report from its
arrival day onward.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ReportingProfile:
    missing_prob: np.ndarray
    delay_prob: np.ndarray
    max_delay: np.ndarray
    spike_prob: np.ndarray
    unreliable: np.ndarray


def make_profile(nominal_reliability, unreliable_fraction, rng, cfg):
    """Reliable sites use their nominal rates; a random subset is made unreliable."""
    N = len(nominal_reliability)
    shortfall = 1 - np.asarray(nominal_reliability)
    unreliable = np.zeros(N, bool)
    n_bad = int(round(unreliable_fraction * N))
    if n_bad:
        unreliable[rng.choice(N, n_bad, replace=False)] = True
    return ReportingProfile(
        missing_prob=np.where(unreliable, cfg.unreliable_missing_prob,
                              shortfall * (1 - cfg.reliable_delay_share)),
        delay_prob=np.where(unreliable, cfg.unreliable_delay_prob,
                            shortfall * cfg.reliable_delay_share),
        max_delay=np.where(unreliable, cfg.unreliable_max_delay, cfg.reliable_max_delay),
        spike_prob=np.where(unreliable, cfg.unreliable_spike_prob, cfg.reliable_spike_prob),
        unreliable=unreliable,
    )


def draw_reporting(profile, T, rng, cfg):
    """Draw arrival days and artefact multipliers. Reusing these for the null and the
    outbreak version of a run keeps the two perfectly paired."""
    N = len(profile.missing_prob)
    u = rng.random((T, N))
    missing = u < profile.missing_prob
    delayed = ~missing & (u < profile.missing_prob + profile.delay_prob)
    delays = 1 + np.floor(rng.random((T, N)) * profile.max_delay).astype(int)
    days = np.arange(T)[:, None]
    arrival = np.where(delayed, days + delays, days)
    arrival[missing] = -1
    arrival[arrival >= T] = -1  # still in transit when the simulation ends
    factor = np.ones((T, N))
    spike = rng.random((T, N)) < profile.spike_prob
    factor[spike] = rng.uniform(cfg.spike_factor_low, cfg.spike_factor_high, spike.sum())
    return arrival.astype(np.int16), factor


def apply_reporting(counts, factor, arrival):
    """Reported counts as they would be received; NaN if never received."""
    reported = np.round(counts * factor)
    reported[arrival < 0] = np.nan
    return reported


def sum_by_arrival(values, arrival):
    """Collapse per-report values indexed by report day onto their arrival day."""
    out = np.zeros_like(values, dtype=float)
    s, i = np.nonzero(arrival >= 0)
    np.add.at(out, (arrival[s, i], i), values[s, i])
    return out


def max_by_arrival(values, arrival):
    """Largest value among the reports reaching each facility on each day; -inf if none."""
    out = np.full(values.shape, -np.inf)
    s, i = np.nonzero((arrival >= 0) & np.isfinite(values))
    np.maximum.at(out, (arrival[s, i], i), values[s, i])
    return out


def count_by_arrival(arrival):
    """Number of reports received per day and facility."""
    return sum_by_arrival(np.ones_like(arrival, dtype=float), arrival)
