"""Static research figures. Colours follow a fixed role per method across all figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from experiments.primary_experiment import FOREST, FULL, METHOD_LABELS
from quorum.hill_activation import hill

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOURS = {FULL: "#2a78d6", FOREST: "#eb6834"}
MARKERS = {FULL: "o", FOREST: "s"}
ORDER = [FULL, FOREST]
SCENARIOS = {"high_reliability": "10% unreliable", "medium_reliability": "30% unreliable",
             "low_reliability": "50% unreliable", "all_scenarios": "All scenarios"}
SEQUENTIAL = LinearSegmentedColormap.from_list("signal", ["#f4f7fb", "#a9cbef", "#2a78d6", "#0d3a73"])


def set_style():
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8, "text.color": INK,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11,
        "axes.titleweight": "semibold", "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "legend.frameon": False, "font.family": "DejaVu Sans", "lines.linewidth": 2,
        "savefig.dpi": 170, "savefig.bbox": "tight"})


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def _draw_edges(ax, world):
    xy = world.facilities[["x_km", "y_km"]].to_numpy()
    i, j = np.nonzero(np.triu(world.W > 0))
    top = world.W.max()
    for a, b in zip(i, j):
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], color="#bdbcb6", lw=0.5 + 1.8 * world.W[a, b] / top, zorder=1)
    return xy


def network_topology(world, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    xy = _draw_edges(ax, world)
    base = world.facilities.baseline_daily_count.to_numpy()
    ax.scatter(xy[:, 0], xy[:, 1], s=26 + base * 2.2, color=COLOURS["full"], edgecolor=SURFACE, lw=1.5, zorder=2)
    for k, (x, y) in enumerate(xy):
        ax.annotate(f"{k}", (x, y), xytext=(0, 8 + np.sqrt(base[k]) * 0.9), textcoords="offset points",
                    ha="center", fontsize=7, color=MUTED)
    for size in (10, 50, 150):
        ax.scatter([], [], s=26 + size * 2.2, color=COLOURS["full"], label=f"{size} symptom reports/day")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, labelspacing=1.2)
    ax.set(xlabel="x (km)", ylabel="y (km)", aspect="equal",
           title=f"Facility network: {len(xy)} sites, {int((world.W > 0).sum() / 2)} distance-weighted links")
    ax.grid(False)
    _save(fig, path)


def signal_propagation(world, bundle, keep, scenario, path):
    r = int(np.nonzero(bundle.scenarios == scenario)[0][0])
    A = keep["full"]["A_out"][r]
    t0, cluster = bundle.t0[r], bundle.cluster_mask[r]
    offsets = (-7, 0, 7, 14, 21, 28)
    vmax = float(np.percentile(A[bundle.t0[r]:bundle.t0[r] + bundle.duration[r]], 99.5))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), sharex=True, sharey=True)
    for ax, off in zip(axes.ravel(), offsets):
        xy = _draw_edges(ax, world)
        points = ax.scatter(xy[:, 0], xy[:, 1], c=A[t0 + off], cmap=SEQUENTIAL, vmin=0, vmax=vmax, s=75,
                            edgecolor=np.where(cluster, INK, "#9c9b95"), lw=np.where(cluster, 1.8, 0.7), zorder=2)
        ax.set_title(f"Outbreak day {off:+d}" if off else "Outbreak start")
        ax.set_aspect("equal")
        ax.grid(False)
    cbar = fig.colorbar(points, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Signal concentration A")
    fig.suptitle(f"Signal propagation in the first pre-specified test run ({SCENARIOS[scenario]}). "
                 "Dark outline: outbreak facility", fontsize=11, fontweight="semibold", x=0.45)
    _save(fig, path)


def hill_curve(params, path):
    fig, ax = plt.subplots(figsize=(6.2, 4))
    A = np.linspace(0, params.K * 5, 500)
    for n, colour, label in ((1.0, "#9c9b95", "n = 1 (no cooperativity)"), (params.n, COLOURS["full"], f"n = {params.n} (model)"),
                             (4.0, "#1baf7a", "n = 4")):
        ax.plot(A, hill(A, params.K, n), color=colour, label=label)
    ax.axvline(params.K, color=MUTED, lw=1, ls="--")
    ax.annotate(f"K = {params.K:g}", (params.K, 0.04), xytext=(5, 0), textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set(xlabel="Signal concentration A", ylabel="Activation H(A)", ylim=(0, 1.02),
           title="Hill activation at the selected half-activation concentration")
    ax.legend(loc="lower right")
    _save(fig, path)


def signal_over_time(bundle, keep, scenario, theta_on, params, path):
    r = int(np.nonzero(bundle.scenarios == scenario)[0][0])
    A = keep["full"]["A_out"][r]
    cluster = bundle.cluster_mask[r]
    t = np.arange(A.shape[0])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(t, A[:, ~cluster], color="#c9c8c2", lw=0.8)
    ax.plot(t, A[:, cluster], color=COLOURS["full"], lw=1.1, alpha=0.65)
    ax.plot(t, A[:, cluster].mean(axis=1), color="#0d3a73", lw=2.4, label="Outbreak facilities, mean")
    ax.plot([], [], color=COLOURS["full"], lw=1.1, label="Outbreak facilities")
    ax.plot([], [], color="#c9c8c2", lw=1, label="Other facilities")
    ax.axvspan(bundle.t0[r], bundle.t0[r] + bundle.duration[r], color="#a9cbef", alpha=0.18, lw=0, label="Outbreak period")
    A_on = params.K * (theta_on / (1 - theta_on)) ** (1 / params.n)
    ax.axhline(A_on, color=INK, lw=1, ls="--", label=f"Activation threshold (A = {A_on:.2f})")
    ax.set(xlabel="Simulation day", ylabel="Signal concentration A",
           title=f"Signal concentration over time ({SCENARIOS[scenario]})", ylim=(0, float(A.max()) * 1.42))
    ax.legend(loc="upper left", ncol=2)
    _save(fig, path)


def network_activation(bundle, keep, scenario, path):
    r = int(np.nonzero(bundle.scenarios == scenario)[0][0])
    H = keep["full"]["H_out"][r]
    cluster = bundle.cluster_mask[r]
    order = np.r_[np.nonzero(cluster)[0], np.nonzero(~cluster)[0]]
    active = keep["full"]["active_out"][r]
    alert = keep["full"]["mask_out"][r]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, height_ratios=[3, 1.2])
    image = a1.imshow(H[:, order].T, aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=max(float(H.max()), 1e-3),
                      interpolation="nearest")
    a1.axhline(cluster.sum() - 0.5, color=INK, lw=1)
    a1.set(ylabel="Facility (outbreak sites on top)", title=f"Network activation over time ({SCENARIOS[scenario]})")
    a1.set_yticks(np.arange(len(order))[::3], order[::3])
    a1.grid(False)
    fig.colorbar(image, ax=a1, pad=0.01).set_label("Hill activation H(A)")
    t = np.arange(H.shape[0])
    a2.fill_between(t, active.sum(axis=1), step="mid", color=COLOURS["full"], alpha=0.25, lw=0, label="Active")
    a2.step(t, alert.sum(axis=1), where="mid", color="#0d3a73", label="Declaring regional alert")
    a2.axvspan(bundle.t0[r], bundle.t0[r] + bundle.duration[r], color="#a9cbef", alpha=0.18, lw=0, label="Outbreak period")
    a2.set(xlabel="Simulation day", ylabel="Facilities", ylim=(0, max(int(active.sum(axis=1).max()), 1) * 1.45))
    a2.legend(loc="upper left", ncol=3)
    _save(fig, path)


def _dot_panel(ax, frame, metric, methods):
    for pos, method in enumerate(methods):
        row = frame[(frame.method == method) & (frame.metric == metric)].iloc[0]
        ax.errorbar(row["mean"], pos, xerr=[[max(row["mean"] - row["ci_low"], 0)], [max(row["ci_high"] - row["mean"], 0)]],
                    fmt=MARKERS[method], color=COLOURS[method], ms=8, mec=SURFACE, mew=1.2, elinewidth=2, capsize=0)
    ax.set_yticks(range(len(methods)), [METHOD_LABELS[m] for m in methods])
    ax.set_ylim(len(methods) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)


def comparison(agg, metric, xlabel, title, path, methods=None, reference=None):
    """Point estimate and 95% CI per scenario. The label on each row identifies the method."""
    methods = methods or ORDER
    frame = agg
    scenarios = list(SCENARIOS)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 0.6 * len(methods) + 1.6), sharex=True, sharey=True)
    for ax, scenario in zip(axes, scenarios):
        _dot_panel(ax, frame[frame.scenario == scenario], metric, methods)
        ax.set_title(SCENARIOS[scenario], fontsize=9.5)
        ax.set_xlabel(xlabel)
        if reference is not None:
            ax.axvline(reference, color=INK, lw=1, ls="--")
    fig.suptitle(title, fontsize=11.5, fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)


def parameter_sensitivity(sweep, path):
    """One-at-a-time sensitivity around the best random-search candidate on validation data."""
    names = [("D", "Diffusion D"), ("delta", "Degradation δ"), ("beta", "Feedback β"), ("K", "Half-activation K"),
             ("quorum", "Quorum fraction q")]
    centre = sweep.attrs["oat_centre"]
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.2), sharey=True)
    for ax, (name, label) in zip(axes, names):
        mask = np.ones(len(sweep), bool)
        for other, value in centre.items():
            if other != name:
                mask &= np.isclose(sweep[other], value)
        frame = sweep[mask].sort_values(name)
        ax.plot(frame[name], frame.val_restricted_delay, color=COLOURS["full"], marker="o", ms=7, mec=SURFACE)
        ax.axvline(centre[name], color=MUTED, lw=1, ls="--")
        infeasible = frame[~frame.feasible.astype(bool)]
        ax.plot(infeasible[name], infeasible.val_restricted_delay, ls="", marker="x", color=INK, ms=8, mew=2)
        ax.set_xlabel(label)
    axes[0].set_ylabel("Restricted delay (days)")
    fig.suptitle("Parameter sensitivity on validation data, lower is better. Dashed: search centre; x: misses alert targets",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)


def bistability_figure(sweeps, map_frame, K, path):
    """Hysteresis loops from a quasi-static production sweep, and the (beta, delta) map."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax, (title, rows) in zip(axes[:2], sweeps.items()):
        for branch, colour, marker in (("up", COLOURS["full"], "o"), ("down", COLOURS[FOREST], "s")):
            part = rows[rows.branch == branch].sort_values("P")
            ax.plot(part.P, part.mean_A, color=colour, marker=marker, ms=4, label=f"Production {branch}")
        ax.set(xlabel="Uniform production P", ylabel="Mean A at steady state", title=title)
        ax.legend()
    pivot = map_frame.pivot(index="delta", columns="beta", values="n_stable")
    sns.heatmap(pivot, ax=axes[2], cmap=SEQUENTIAL, annot=True, fmt=".0f", cbar=False, linewidths=2, linecolor=SURFACE)
    axes[2].invert_yaxis()
    axes[2].set(title=f"Stable states of one node, P = 0, K = {K:g}", xlabel="Feedback β", ylabel="Degradation δ")
    fig.tight_layout()
    _save(fig, path)


def tradeoff_figure(curves, target, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharex=True)
    pooled = curves[(curves.scenario == "all_scenarios") & curves.usable_for_matching]
    for method in ORDER:
        frame = pooled[pooled.method == method].sort_values("false_alerts_per_year")
        for ax, col in zip(axes, ("attributable_detection_rate", "attributable_restricted_delay_days")):
            ax.plot(frame.false_alerts_per_year, frame[col], color=COLOURS[method], marker=MARKERS[method],
                    ms=5, lw=1.8, label=METHOD_LABELS[method])
    for ax in axes:
        ax.axvline(target, color=INK, lw=1, ls="--")
        ax.set_xlim(0, 8)
        ax.set_xlabel("False alerts per year (test null runs)")
    axes[0].set(ylabel="Attributable detection rate", title="Detection versus false alerts")
    axes[1].set(ylabel="Attributable restricted delay (days)", title="Delay versus false alerts")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    _save(fig, path)
