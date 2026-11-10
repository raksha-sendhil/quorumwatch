"""Facility activation with hysteresis and the local quorum decision."""
import numpy as np


def apply_hysteresis(H, theta_on, theta_off):
    """Latch ON at H >= theta_on, release only below theta_off. Shape (..., T, N)."""
    active = np.zeros(H.shape, dtype=bool)
    state = np.zeros(H.shape[:-2] + H.shape[-1:], dtype=bool)
    for t in range(H.shape[-2]):
        Ht = H[..., t, :]
        state = (state & (Ht >= theta_off)) | (Ht >= theta_on)
        active[..., t, :] = state
    return active


def quorum_alert(active, closed_neighbourhood, quorum, min_count):
    """A facility declares a regional alert when it is active and at least a fraction
    `quorum` of its closed neighbourhood (and at least `min_count` sites) is active.
    Each facility evaluates this from its own state and its neighbours' state bits."""
    Nb = closed_neighbourhood.astype(np.float32)
    counts = active.astype(np.float32) @ Nb
    sizes = Nb.sum(axis=1)
    return active & (counts >= min_count) & (counts / sizes >= quorum - 1e-9)
