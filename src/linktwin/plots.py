"""Figures (earth-tone palette, 200 dpi)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BROWN, TEAL, OCHRE, TERRA, OLIVE, TAUPE, SAND = (
    "#6b4226", "#2a7f7f", "#c8902e", "#a4492f", "#5f7a45", "#8a8070", "#e9dcc3",
)
EARTH_DIV = LinearSegmentedColormap.from_list("earth_div", [TERRA, "#e2b98a", SAND, "#7fb5ac", TEAL])
DPI = 200


def _style() -> None:
    plt.rcParams.update({"axes.facecolor": "#fbf8f1", "figure.facecolor": "white", "axes.edgecolor": BROWN,
                         "axes.labelcolor": BROWN, "xtick.color": BROWN, "ytick.color": BROWN,
                         "axes.titlecolor": BROWN, "font.size": 10})


def fig_ecdf(cycle_eff: dict[str, list[float]], colors: dict[str, str], out: Path) -> None:
    """Empirical CDF of per-cycle delivered / oracle value for each method."""
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for name, vals in cycle_eff.items():
        x = np.sort(np.asarray(vals))
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=name, color=colors[name], lw=2)
    ax.set_xlabel("per-cycle delivered value / oracle value (dimensionless)")
    ax.set_ylabel("empirical CDF (fraction of cycles)")
    ax.set_title("Per-cycle efficiency distribution at the headline operating point")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def fig_violin(df: pd.DataFrame, order: list[str], labels: dict[str, str], colors: dict[str, str], out: Path) -> None:
    """Violin + box of per-seed efficiency with the individual seeds overlaid."""
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    data = [df[df.method == m].efficiency.to_numpy() for m in order]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, m in zip(parts["bodies"], order):
        body.set_facecolor(colors[m]); body.set_alpha(0.35); body.set_edgecolor(colors[m])
    bp = ax.boxplot(data, widths=0.25, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set(facecolor="white", edgecolor=BROWN)
    for med in bp["medians"]:
        med.set(color=BROWN, lw=2)
    rng = np.random.default_rng(0)
    for i, (m, d) in enumerate(zip(order, data), start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=16, color=colors[m], edgecolor=BROWN, lw=0.4, zorder=3)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([labels[m] for m in order], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("delivered value / oracle value per seed (dimensionless)")
    ax.set_title("Seed-level efficiency: baselines, digital twin and twin ablations")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def fig_phase(grid: pd.DataFrame, out: Path, quick: bool = False) -> None:
    """Filled-contour map of twin minus strongest baseline over volatility x pair budget."""
    _style()
    piv = grid.pivot(index="cfg_n_pairs", columns="drift_sigma", values="mean_diff")
    xs, ys = piv.columns.to_numpy(), piv.index.to_numpy()
    z = piv.to_numpy()
    lim = max(abs(z).max(), 1e-3)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    levels = np.linspace(-lim, lim, 13)
    cs = ax.contourf(xs, ys, z, levels=levels, cmap=EARTH_DIV)
    if z.min() < 0 < z.max():
        ax.contour(xs, ys, z, levels=[0.0], colors=BROWN, linewidths=2)
    ax.scatter(np.repeat(xs[None, :], len(ys), 0).ravel(), np.repeat(ys[:, None], len(xs), 1).ravel(),
               s=8, color=BROWN)
    ax.set_yscale("log", base=2)
    ax.set_yticks(ys); ax.set_yticklabels([str(int(y)) for y in ys])
    ax.set_xlabel("drift volatility of the fidelity coordinate, sigma (per cycle, logit units)")
    ax.set_ylabel("raw pairs per cycle, N")
    ax.set_title("Twin minus strongest baseline (efficiency difference)")
    fig.colorbar(cs, label="mean efficiency difference (teal: twin ahead)")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def fig_robust(rob: pd.DataFrame, out: Path) -> None:
    """Twin advantage over the strongest baseline under three perturbations."""
    _style()
    panels = [("gate_noise", "local gate depolarising noise per round, q", [("aware", TEAL)]),
              ("readout", "per-qubit readout error, e", [("aware", TEAL), ("unaware", TERRA)]),
              ("drift_scale", "assumed / true drift volatility", [("aware", TEAL)])]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for ax, (kind, xlabel, series) in zip(axes, panels):
        for tag, col in series:
            d = rob[(rob.kind == kind) & (rob.tag == tag)]
            if d.empty:
                continue
            g = d.groupby("x").mean_diff.agg(["mean", "std"]).reset_index()
            lab = {"aware": "estimators know the setting", "unaware": "estimators assume e = 0"}[tag]
            ax.errorbar(g.x, g["mean"], yerr=g["std"].fillna(0), color=col, marker="o", capsize=3, lw=2, label=lab)
            ax.scatter(d.x, d.mean_diff, s=12, color=col, alpha=0.5)
        ax.axhline(0, color=BROWN, lw=1, ls="--")
        ax.set_xlabel(xlabel)
        if kind == "drift_scale":
            ax.set_xscale("log", base=2)
        if len(series) > 1 or kind != "gate_noise":
            ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("twin minus strongest baseline (efficiency)")
    fig.suptitle("Robustness of the twin advantage (mean and std over seeds; dots are seeds)", color=BROWN)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
