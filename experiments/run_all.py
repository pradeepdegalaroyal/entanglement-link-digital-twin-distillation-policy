"""Reproduce every result and figure.  ``python experiments/run_all.py`` (full) or ``--quick`` (smoke run)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from linktwin import plots  # noqa: E402
from linktwin.analysis import best_baseline, best_of_family, paired_summary, per_seed  # noqa: E402
from linktwin.drift import DriftModel  # noqa: E402
from linktwin.methods import SHADOW_GRID, LinkConfig, TwinConfig  # noqa: E402
from linktwin.study import evaluate  # noqa: E402

TWINS = {
    "twin": TwinConfig(),
    "twin_no_memory": TwinConfig(use_memory=False),
    "twin_fixed_monitoring": TwinConfig(adaptive=False, m_min=12),
    "twin_no_min_guard": TwinConfig(m_min=0),
    "twin_no_jump_model": TwinConfig(jumps_in_model=False),
}


def _task(args: tuple) -> list[dict]:
    drift, cfg, seed, twins, grid, keep, meta = args
    rows = evaluate(drift, cfg, seed, twins, grid, keep_cycles=keep)
    for r in rows:
        r.update(meta)
    return rows


def _map(tasks: list[tuple], workers: int) -> list[dict]:
    if workers <= 1:
        out = [_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers) as ex:
            out = list(ex.map(_task, tasks))
    return [r for rows in out for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke run (< 1 minute)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    q = args.quick
    if q:
        args.workers = 1
    t0 = time.time()
    res, fig = ROOT / "results", ROOT / "figures"
    res.mkdir(exist_ok=True)
    fig.mkdir(exist_ok=True)

    n_cycles = 30 if q else 150
    n_seeds = 2 if q else 10
    grid_seeds = 2 if q else 5
    grid_cycles = 30 if q else 100
    train = [100] if q else [100, 101, 102]
    shadow_grid = [(1, 24), (8, 48)] if q else SHADOW_GRID
    twins = TWINS if not q else {k: replace(v, n_particles=120, n_preposterior=16) for k, v in TWINS.items()}
    tw_only = {"twin": twins["twin"]}
    drift0 = DriftModel()
    cfg0 = LinkConfig(n_cycles=n_cycles)
    summary: dict = {"quick": q, "headline_config": {"drift_sigma": drift0.sigma, **cfg0.__dict__}}

    # ---- 1. headline operating point (+ ablations) ----------------------------------------------
    print("[1/4] headline + ablation", flush=True)
    rows = _map([(drift0, cfg0, s, twins, shadow_grid, True, {}) for s in range(n_seeds)], args.workers)
    cyc: dict[str, list[float]] = {}
    for r in rows:
        cyc.setdefault(r["method"], []).extend(r.pop("cycle_eff", []))
    head = pd.DataFrame(rows)
    head.to_csv(res / "headline.csv", index=False)
    # train-tuned shadow: choose (P, M) on disjoint seeds, then evaluate on the test seeds
    tr = pd.DataFrame(_map([(drift0, cfg0, s, {}, shadow_grid, False, {}) for s in train], args.workers))
    tuned_name, _ = best_of_family(tr[tr.method.str.startswith("shadow_")], "shadow_")
    shadow_best_name, _ = best_of_family(head, "shadow_")
    fixed_best_name, _ = best_of_family(head, "fixed_")
    strongest_name, _ = best_baseline(head)
    summary["selected"] = {"shadow_tuned_on_train": tuned_name, "shadow_hindsight": shadow_best_name,
                           "fixed_hindsight": fixed_best_name, "strongest_baseline": strongest_name}
    table = {}
    entries = {"oracle": "oracle", "raw (no distillation)": "fixed_raw", "fixed DEJMPS x2": "fixed_dejmpsx2",
               "best fixed action (hindsight)": fixed_best_name, "static digital model": "static_model",
               "digital shadow, tuned on held-out seeds": tuned_name,
               "digital shadow, best in hindsight": shadow_best_name, "digital twin": "twin"}
    for label, m in entries.items():
        e = per_seed(head, m)
        table[label] = {"method_id": m, "efficiency_mean": float(e.mean()), "efficiency_std": float(e.std(ddof=1)),
                        "monitor_pairs_mean": float(per_seed(head, m, "mean_monitor_pairs").mean()),
                        "outage_rate_mean": float(per_seed(head, m, "outage_rate").mean())}
    summary["headline_table"] = table
    twin_e = per_seed(head, "twin")
    summary["twin_vs"] = {label: paired_summary(twin_e, per_seed(head, m)) for label, m in entries.items()
                          if label not in ("oracle", "digital twin")}
    summary["ablation"] = {}
    for name in twins:
        e = per_seed(head, name)
        summary["ablation"][name] = {
            "efficiency_mean": float(e.mean()), "efficiency_std": float(e.std(ddof=1)),
            "monitor_pairs_mean": float(per_seed(head, name, "mean_monitor_pairs").mean()),
            "vs_full_twin": paired_summary(e, twin_e)}
    keep = {"twin", tuned_name, "fixed_dejmpsx2", "static_model", "fixed_raw", fixed_best_name}
    (res / "cycle_efficiency.json").write_text(json.dumps({k: v for k, v in cyc.items() if k in keep}))

    # ---- 2. phase diagram: drift volatility x pair budget ----------------------------------------
    print("[2/4] phase grid", flush=True)
    sigmas = [0.05, 0.2, 0.4] if q else [0.05, 0.1, 0.2, 0.3, 0.4]
    budgets = [48, 192] if q else [24, 48, 96, 192, 384]
    tasks = [(replace(drift0, sigma=sg), LinkConfig(n_pairs=nb, n_cycles=grid_cycles), s, {"twin": tw_only["twin"]},
              shadow_grid, False, {"drift_sigma": sg, "cfg_n_pairs": nb})
             for sg in sigmas for nb in budgets for s in range(grid_seeds)]
    gdf = pd.DataFrame(_map(tasks, args.workers))
    gdf.to_csv(res / "phase_grid_raw.csv", index=False)
    out = []
    for (sg, nb), d in gdf.groupby(["drift_sigma", "cfg_n_pairs"]):
        name, base = best_baseline(d)
        ps = paired_summary(per_seed(d, "twin"), base)
        out.append({"drift_sigma": sg, "cfg_n_pairs": nb, "strongest_baseline": name,
                    "twin_eff": float(per_seed(d, "twin").mean()), "baseline_eff": float(base.mean()),
                    "twin_monitor_pairs": float(per_seed(d, "twin", "mean_monitor_pairs").mean()), **ps})
    phase = pd.DataFrame(out)
    phase.to_csv(res / "phase_grid.csv", index=False)

    # ---- 3. robustness sweeps -------------------------------------------------------------------
    print("[3/4] robustness", flush=True)
    specs = []
    gates = [0.0, 0.04] if q else [0.0, 0.01, 0.02, 0.04, 0.08]
    ros = [0.0, 0.1] if q else [0.0, 0.02, 0.05, 0.1]
    scales = [0.5, 2.0] if q else [0.25, 0.5, 1.0, 2.0, 4.0]
    for g in gates:
        specs.append(("gate_noise", "aware", g, LinkConfig(gate_noise=g, n_cycles=grid_cycles), tw_only["twin"]))
    for e in ros:
        specs.append(("readout", "aware", e,
                      LinkConfig(readout_error=e, assumed_readout_error=e, n_cycles=grid_cycles), tw_only["twin"]))
        specs.append(("readout", "unaware", e,
                      LinkConfig(readout_error=e, assumed_readout_error=0.0, n_cycles=grid_cycles), tw_only["twin"]))
    for sc in scales:
        specs.append(("drift_scale", "aware", sc, LinkConfig(n_cycles=grid_cycles),
                      replace(tw_only["twin"], drift_scale=sc)))
    rob_tasks = [(drift0, cfg, s, {"twin": tw}, shadow_grid, False, {"kind": kind, "tag": tag, "x": x})
                 for kind, tag, x, cfg, tw in specs for s in range(grid_seeds)]
    rdf = pd.DataFrame(_map(rob_tasks, args.workers))
    rrows = []
    for (kind, tag, x), d in rdf.groupby(["kind", "tag", "x"]):
        name, base = best_baseline(d)
        tw_e = per_seed(d, "twin")
        for s in tw_e.index:
            rrows.append({"kind": kind, "tag": tag, "x": x, "seed": s, "twin_eff": tw_e[s],
                          "baseline_eff": base[s], "strongest_baseline": name, "mean_diff": tw_e[s] - base[s]})
    rob = pd.DataFrame(rrows)
    rob.to_csv(res / "robustness.csv", index=False)

    # ---- 4. figures + summary --------------------------------------------------------------------
    print("[4/4] figures", flush=True)
    cols = {"twin": plots.TEAL, tuned_name: plots.OCHRE, "fixed_dejmpsx2": plots.BROWN,
            "static_model": plots.TAUPE, "fixed_raw": plots.TERRA, fixed_best_name: plots.OLIVE}
    names = {"twin": "digital twin", tuned_name: f"digital shadow ({tuned_name.replace('shadow_', '')})",
             "fixed_dejmpsx2": "fixed DEJMPS x2", "static_model": "static digital model",
             "fixed_raw": "raw pairs (no distillation)", fixed_best_name: f"best fixed action ({fixed_best_name[6:]})"}
    plots.fig_ecdf({names[k]: cyc[k] for k in cols if k in cyc}, {names[k]: cols[k] for k in cols if k in cyc},
                   fig / "fig1_ecdf_cycle_efficiency.png")
    order = ["fixed_raw", "static_model", tuned_name, "fixed_dejmpsx2", "twin_no_memory", "twin_fixed_monitoring",
             "twin_no_min_guard", "twin_no_jump_model", "twin"]
    lab = {"fixed_raw": "raw pairs", "static_model": "static model", tuned_name: "shadow (tuned)",
           "fixed_dejmpsx2": "fixed DEJMPS x2", "twin_no_memory": "twin: no filter memory",
           "twin_fixed_monitoring": "twin: fixed monitoring", "twin_no_min_guard": "twin: no minimum guard",
           "twin_no_jump_model": "twin: no jump model", "twin": "digital twin (full)"}
    pal = [plots.TERRA, plots.TAUPE, plots.OCHRE, plots.BROWN, plots.OLIVE, plots.OLIVE, plots.OLIVE, plots.OLIVE,
           plots.TEAL]
    plots.fig_violin(head, order, lab, dict(zip(order, pal)), fig / "fig2_seed_efficiency_violin.png")
    plots.fig_phase(phase, fig / "fig3_phase_twin_vs_baseline.png", q)
    plots.fig_robust(rob, fig / "fig4_robustness.png")

    summary["phase_grid"] = phase.to_dict(orient="records")
    summary["twin_ahead_cells"] = int((phase.mean_diff > 0).sum())
    summary["n_cells"] = int(len(phase))
    summary["robustness"] = {
        f"{k}|{t}|{x}": {"mean_diff": float(d.mean_diff.mean()),
                         "std_diff": float(d.mean_diff.std(ddof=1)) if len(d) > 1 else 0.0}
        for (k, t, x), d in rob.groupby(["kind", "tag", "x"])}
    summary["seeds"] = {"headline": n_seeds, "grid": grid_seeds, "tuning": len(train)}
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    (res / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done in {summary['runtime_seconds']} s")


if __name__ == "__main__":
    main()
