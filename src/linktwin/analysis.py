"""Aggregation helpers: hindsight-best and train-tuned baselines, paired statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_seed(df: pd.DataFrame, method: str, col: str = "efficiency") -> pd.Series:
    """Efficiency indexed by seed for one method."""
    d = df[df.method == method]
    return d.set_index("seed")[col].sort_index()


def best_of_family(df: pd.DataFrame, prefix: str, col: str = "efficiency") -> tuple[str, pd.Series]:
    """Configuration of a baseline family with the highest mean over the seeds of ``df`` (hindsight)."""
    fam = df[df.method.str.startswith(prefix)]
    means = fam.groupby("method")[col].mean()
    name = str(means.idxmax())
    return name, per_seed(df, name, col)


def best_baseline(df: pd.DataFrame) -> tuple[str, pd.Series]:
    """Strongest non-twin baseline in hindsight over all fixed policies and shadow configurations."""
    cand = df[df.method.str.startswith(("fixed_", "shadow_", "static_"))]
    name = str(cand.groupby("method").efficiency.mean().idxmax())
    return name, per_seed(df, name)


def paired_summary(a: pd.Series, b: pd.Series, n_boot: int = 10000, seed: int = 0) -> dict:
    """Mean paired difference ``a - b`` over seeds with a bootstrap 95% interval and win count."""
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d))).mean(axis=1)
    return {
        "mean_diff": float(d.mean()),
        "std_diff": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "wins": int((d > 0).sum()),
        "n": int(len(d)),
    }
