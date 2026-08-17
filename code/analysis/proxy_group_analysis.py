#!/usr/bin/env python3
"""Single-seed atlas-proxy analysis for matched, control, and ablation runs."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind

from morphometry import MUSCLE_LABELS, extract_mask_morphometry


def normalize_id(value: object) -> str:
    text = re.sub(r"\.0$", "", str(value).strip())
    return text.zfill(4) if text.isdigit() else text


def bh_adjust(values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return q
    order = valid[np.argsort(p[valid])]
    adjusted = p[order] * valid.size / np.arange(1, valid.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return q


def safe_rho(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 5 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan, math.nan
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def bootstrap_rho(x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, x.size, x.size)
        rho, _ = safe_rho(x[idx], y[idx])
        if np.isfinite(rho):
            values.append(rho)
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5])) if values else (math.nan, math.nan)


def bootstrap_gap(higher: np.ndarray, lower: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        values[i] = rng.choice(higher, higher.size, replace=True).mean() - rng.choice(lower, lower.size, replace=True).mean()
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def permutation_gap_p(higher: np.ndarray, lower: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    observed = abs(float(higher.mean() - lower.mean()))
    pooled = np.concatenate([higher, lower])
    higher_n = higher.size
    exceed = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(pooled)
        gap = abs(float(shuffled[:higher_n].mean() - shuffled[higher_n:].mean()))
        exceed += int(gap >= observed)
    return (exceed + 1.0) / (n_perm + 1.0)


def hedges_g(higher: np.ndarray, lower: np.ndarray) -> float:
    n1, n0 = higher.size, lower.size
    pooled_var = ((n1 - 1) * higher.var(ddof=1) + (n0 - 1) * lower.var(ddof=1)) / (n1 + n0 - 2)
    if pooled_var <= 0:
        return math.nan
    d = (higher.mean() - lower.mean()) / math.sqrt(pooled_var)
    correction = 1.0 - 3.0 / (4.0 * (n1 + n0) - 9.0)
    return float(correction * d)


def prepare(path: str, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"file_id": str})
    frame["merge_id"] = frame["file_id"].map(normalize_id)
    rename = {
        column: f"{prefix}_{column}"
        for column in frame.columns
        if column not in {"file_id", "merge_id"} and (column.endswith("_mcsa") or column.endswith("_vol"))
    }
    return frame.rename(columns=rename)


def analyze_one(
    merged: pd.DataFrame,
    muscle: str,
    measure: str,
    atlas_value: float,
    n_total: int,
    n_boot: int,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[dict[str, object], pd.DataFrame]:
    gen_col = f"gen_{muscle}_{measure}"
    gt_col = f"gt_{muscle}_{measure}"
    sub = merged[["merge_id", gen_col, gt_col]].copy()
    sub[gen_col] = pd.to_numeric(sub[gen_col], errors="coerce")
    sub[gt_col] = pd.to_numeric(sub[gt_col], errors="coerce")
    sub = sub.dropna()
    gen = sub[gen_col].to_numpy(dtype=float)
    gt = sub[gt_col].to_numpy(dtype=float)
    delta = (gen - atlas_value) / atlas_value * 100.0
    gt_delta = (gt - atlas_value) / atlas_value * 100.0

    # The manuscript's groups are morphology-defined within the metric-specific
    # complete-case subset; they are not clinical severity categories.
    lower_threshold = float(np.quantile(gt, 0.30))
    higher_threshold = float(np.quantile(gt, 0.70))
    lower_mask = gt <= lower_threshold
    higher_mask = gt >= higher_threshold
    lower = delta[lower_mask]
    higher = delta[higher_mask]
    gap = float(higher.mean() - lower.mean())
    t_p = float(ttest_ind(higher, lower, equal_var=False).pvalue)
    rho, rho_p = safe_rho(gen, gt)
    rho_lo, rho_hi = bootstrap_rho(gen, gt, n_boot, rng)
    gap_lo, gap_hi = bootstrap_gap(higher, lower, n_boot, rng)
    perm_p = permutation_gap_p(higher, lower, n_perm, rng)
    diff = gen - gt
    ape = np.abs(diff / np.where(gt == 0, np.nan, gt)) * 100.0
    direction = np.sign(gen - atlas_value) == np.sign(gt - atlas_value)

    details = pd.DataFrame({
        "file_id": sub["merge_id"],
        "muscle": muscle,
        "measure": measure,
        "generated": gen,
        "ground_truth": gt,
        "atlas_reference": atlas_value,
        "generated_delta_from_atlas_percent": delta,
        "gt_delta_from_atlas_percent": gt_delta,
        "reference_group": np.where(higher_mask, "higher", np.where(lower_mask, "lower", "middle")),
        "absolute_error": np.abs(diff),
        "absolute_percentage_error": ape,
        "direction_agreement_vs_atlas": direction,
    })
    row = {
        "muscle": muscle,
        "measure": measure,
        "n_total": n_total,
        "n_valid": int(len(sub)),
        "ssr_percent": len(sub) / n_total * 100.0,
        "atlas_reference": atlas_value,
        "lower_group_n": int(lower.size),
        "higher_group_n": int(higher.size),
        "lower_reference_threshold": lower_threshold,
        "higher_reference_threshold": higher_threshold,
        "higher_minus_lower_delta_gap_pp": gap,
        "gap_bootstrap_ci95_low_pp": gap_lo,
        "gap_bootstrap_ci95_high_pp": gap_hi,
        "welch_t_p": t_p,
        "permutation_p": perm_p,
        "hedges_g": hedges_g(higher, lower),
        "spearman_rho": rho,
        "spearman_bootstrap_ci95_low": rho_lo,
        "spearman_bootstrap_ci95_high": rho_hi,
        "spearman_p": rho_p,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mape_percent": float(np.nanmean(ape)),
        "mean_signed_error": float(np.mean(diff)),
        "direction_agreement_percent": float(np.mean(direction) * 100.0),
    }
    return row, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-metrics-csv", required=True)
    parser.add_argument("--gt-metrics-csv", required=True)
    parser.add_argument("--atlas-mask", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    gen = prepare(args.generated_metrics_csv, "gen")
    gt = prepare(args.gt_metrics_csv, "gt")
    merged = gen.merge(gt, on="merge_id", how="inner", validate="one_to_one")
    atlas = extract_mask_morphometry(args.atlas_mask, file_id="fixed_atlas")
    n_total = int(gt["merge_id"].nunique())

    rows: list[dict[str, object]] = []
    details: list[pd.DataFrame] = []
    counter = 0
    for measure in ("mcsa", "vol"):
        for muscle in MUSCLE_LABELS:
            row, detail = analyze_one(
                merged, muscle, measure, float(atlas[f"{muscle}_{measure}"]), n_total,
                args.bootstrap, args.permutations, np.random.default_rng(args.seed + counter),
            )
            row["condition"] = args.condition
            detail.insert(0, "condition", args.condition)
            rows.append(row)
            details.append(detail)
            counter += 1

    metrics = pd.DataFrame(rows)
    for measure, index in metrics.groupby("measure", sort=False).groups.items():
        metrics.loc[index, "spearman_bh_q"] = bh_adjust(metrics.loc[index, "spearman_p"])
        metrics.loc[index, "welch_t_bh_q"] = bh_adjust(metrics.loc[index, "welch_t_p"])
        metrics.loc[index, "permutation_bh_q"] = bh_adjust(metrics.loc[index, "permutation_p"])
    metrics["spearman_bh_significant_0_05"] = metrics["spearman_bh_q"] < 0.05
    metrics["gap_permutation_bh_significant_0_05"] = metrics["permutation_bh_q"] < 0.05

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(f"{prefix}_muscle_metrics.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(f"{prefix}_per_case.csv", index=False)
    merged.to_csv(f"{prefix}_merged.csv", index=False)
    pd.DataFrame([atlas]).to_csv(f"{prefix}_atlas_metrics.csv", index=False)
    print(metrics.to_string(index=False))
    print("[Done]", prefix)


if __name__ == "__main__":
    main()
