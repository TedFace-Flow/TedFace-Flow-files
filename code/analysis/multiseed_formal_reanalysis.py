#!/usr/bin/env python3
"""JMS multi-seed morphology-reference and patient-correspondence analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


MUSCLES = (
    "medial_rectus_right",
    "inferior_rectus_right",
    "superior_rectus_right",
    "lateral_rectus_right",
    "medial_rectus_left",
    "inferior_rectus_left",
    "superior_rectus_left",
    "lateral_rectus_left",
)
MEASURES = ("mcsa", "vol")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-2026-csv", required=True)
    parser.add_argument("--seed-42-csv", required=True)
    parser.add_argument("--seed-1024-csv", required=True)
    parser.add_argument("--seedmean-merged-csv", required=True)
    parser.add_argument("--atlas-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--group-permutations", type=int, default=10000)
    parser.add_argument("--pairing-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def normalize_id(value: object) -> str:
    text = re.sub(r"\.0$", "", str(value).strip())
    return text.zfill(4) if text.isdigit() else text


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_adjust(values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if not valid.size:
        return q
    order = valid[np.argsort(p[valid])]
    adjusted = p[order] * valid.size / np.arange(1, valid.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return q


def metric_names() -> list[str]:
    return [f"{muscle}_{measure}" for measure in MEASURES for muscle in MUSCLES]


def read_seed_metrics(path: Path, expected_ids: set[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"file_id": str})
    if "file_id" not in frame:
        raise ValueError(f"Missing file_id: {path}")
    frame["file_id"] = frame["file_id"].map(normalize_id)
    if frame["file_id"].isna().any() or frame["file_id"].duplicated().any():
        raise ValueError(f"file_id must be non-null and unique: {path}")
    missing_columns = sorted(set(metric_names()) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing metric columns in {path}: {missing_columns}")
    for column in metric_names():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.set_index("file_id").sort_index()
    if expected_ids is not None and set(frame.index) != expected_ids:
        raise ValueError(f"Patient ID set mismatch: {path}")
    return frame


def read_seedmean_merged(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, dtype={"merge_id": str, "file_id_x": str, "file_id_y": str})
    if "merge_id" not in frame:
        raise ValueError("seedmean merged CSV must contain merge_id")
    for column in ("merge_id", "file_id_x", "file_id_y"):
        if column in frame:
            frame[column] = frame[column].map(normalize_id)
    if frame["merge_id"].isna().any() or frame["merge_id"].duplicated().any():
        raise ValueError("merge_id must be non-null and unique")
    for column in ("file_id_x", "file_id_y"):
        if column in frame and not (frame[column] == frame["merge_id"]).all():
            raise ValueError(f"{column} does not match merge_id")
    gt_columns = [f"gt_{name}" for name in metric_names()]
    gen_columns = [f"gen_{name}" for name in metric_names()]
    missing_columns = sorted(set(gt_columns + gen_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing GT columns: {missing_columns}")
    result = frame[["merge_id", *gt_columns]].copy()
    for column in gt_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.set_index("merge_id").sort_index()
    if result.isna().any().any():
        missing = result.isna().sum()
        raise ValueError(f"Ground-truth metrics contain missing values: {missing[missing > 0].to_dict()}")
    generated = frame[["merge_id", *gen_columns]].copy()
    for column in gen_columns:
        generated[column] = pd.to_numeric(generated[column], errors="coerce")
    generated = generated.rename(columns={f"gen_{name}": name for name in metric_names()})
    generated = generated.set_index("merge_id").sort_index()
    return result, generated


def read_atlas(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError("Atlas CSV must contain exactly one row")
    atlas: dict[str, float] = {}
    for metric in metric_names():
        value = float(frame.iloc[0][metric])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid atlas value for {metric}: {value}")
        atlas[metric] = value
    return atlas


def build_rule_frames(
    seeds: dict[int, pd.DataFrame], gt: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient_ids = gt.index
    rule_3of3 = pd.DataFrame(index=patient_ids)
    rule_2of3 = pd.DataFrame(index=patient_ids)
    audit_rows: list[dict[str, object]] = []

    for metric in metric_names():
        values = np.column_stack(
            [seeds[seed].loc[patient_ids, metric].to_numpy(dtype=float) for seed in (2026, 42, 1024)]
        )
        valid = np.isfinite(values)
        valid_count = valid.sum(axis=1)
        sums = np.where(valid, values, 0.0).sum(axis=1)
        mean_available = np.divide(
            sums,
            valid_count,
            out=np.full(patient_ids.size, np.nan, dtype=float),
            where=valid_count > 0,
        )
        rule_3of3[metric] = np.where(valid_count == 3, mean_available, np.nan)
        rule_2of3[metric] = np.where(valid_count >= 2, mean_available, np.nan)

        gt_values = gt[f"gt_{metric}"].to_numpy(dtype=float)
        lower_all = float(np.quantile(gt_values, 0.30))
        higher_all = float(np.quantile(gt_values, 0.70))
        full_reference_group = np.where(
            gt_values <= lower_all,
            "lower",
            np.where(gt_values >= higher_all, "higher", "middle"),
        )

        for index, patient_id in enumerate(patient_ids):
            missing_seeds = [str(seed) for pos, seed in enumerate((2026, 42, 1024)) if not valid[index, pos]]
            audit_rows.append(
                {
                    "file_id": patient_id,
                    "metric": metric,
                    "muscle": metric.rsplit("_", 1)[0],
                    "measure": metric.rsplit("_", 1)[1],
                    "seed_2026": values[index, 0],
                    "seed_42": values[index, 1],
                    "seed_1024": values[index, 2],
                    "valid_seed_count": int(valid_count[index]),
                    "missing_seeds": ";".join(missing_seeds),
                    "rule_3of3_value": rule_3of3.iloc[index][metric],
                    "rule_atleast2of3_value": rule_2of3.iloc[index][metric],
                    "gt_value": gt_values[index],
                    "full_74_reference_group": full_reference_group[index],
                }
            )

    audit = pd.DataFrame(audit_rows)
    patient_audit = (
        audit.groupby("file_id", as_index=False)
        .agg(
            metric_count=("metric", "size"),
            metrics_valid_3of3=("valid_seed_count", lambda x: int((x == 3).sum())),
            metrics_valid_atleast2of3=("valid_seed_count", lambda x: int((x >= 2).sum())),
            metrics_with_only_one_seed=("valid_seed_count", lambda x: int((x == 1).sum())),
            metrics_with_zero_seeds=("valid_seed_count", lambda x: int((x == 0).sum())),
        )
    )
    rule_3of3.index.name = "file_id"
    rule_2of3.index.name = "file_id"
    return rule_3of3, rule_2of3, audit, patient_audit


def advance_rng_like_rho_bootstrap(n: int, repetitions: int, rng: np.random.Generator) -> None:
    for _ in range(repetitions):
        rng.integers(0, n, n)


def bootstrap_gap(
    higher: np.ndarray, lower: np.ndarray, repetitions: int, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        values[index] = (
            rng.choice(higher, higher.size, replace=True).mean()
            - rng.choice(lower, lower.size, replace=True).mean()
        )
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def group_permutation_p(
    higher: np.ndarray, lower: np.ndarray, repetitions: int, rng: np.random.Generator
) -> float:
    observed = abs(float(higher.mean() - lower.mean()))
    pooled = np.concatenate([higher, lower])
    higher_n = higher.size
    exceed = 0
    for _ in range(repetitions):
        shuffled = rng.permutation(pooled)
        gap = abs(float(shuffled[:higher_n].mean() - shuffled[higher_n:].mean()))
        exceed += int(gap >= observed)
    return float((exceed + 1.0) / (repetitions + 1.0))


def hedges_g(higher: np.ndarray, lower: np.ndarray) -> float:
    n1, n0 = higher.size, lower.size
    if n1 < 2 or n0 < 2:
        return math.nan
    pooled_var = (
        (n1 - 1) * higher.var(ddof=1) + (n0 - 1) * lower.var(ddof=1)
    ) / (n1 + n0 - 2)
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        return math.nan
    correction = 1.0 - 3.0 / (4.0 * (n1 + n0) - 9.0)
    return float(correction * (higher.mean() - lower.mean()) / math.sqrt(pooled_var))


def analyze_rule(
    rule_name: str,
    generated: pd.DataFrame,
    gt: pd.DataFrame,
    atlas: dict[str, float],
    bootstrap_repetitions: int,
    group_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    counter = 0
    for measure in MEASURES:
        for muscle in MUSCLES:
            metric = f"{muscle}_{measure}"
            gen = generated[metric].to_numpy(dtype=float)
            gt_values = gt[f"gt_{metric}"].to_numpy(dtype=float)
            valid = np.isfinite(gen) & np.isfinite(gt_values)
            gen_valid = gen[valid]
            gt_valid = gt_values[valid]
            if gen_valid.size < 5:
                raise ValueError(f"Too few valid cases for {rule_name} {metric}")

            # Groups are defined from reference CT measurements in the
            # metric-specific valid subset for this missingness rule.
            lower_threshold = float(np.quantile(gt_valid, 0.30))
            higher_threshold = float(np.quantile(gt_valid, 0.70))
            delta = (gen_valid - atlas[metric]) / atlas[metric] * 100.0
            lower = delta[gt_valid <= lower_threshold]
            higher = delta[gt_valid >= higher_threshold]
            gap = float(higher.mean() - lower.mean())
            welch_p = float(ttest_ind(higher, lower, equal_var=False).pvalue)

            rng = np.random.default_rng(seed + counter)
            advance_rng_like_rho_bootstrap(gen_valid.size, bootstrap_repetitions, rng)
            ci_low, ci_high = bootstrap_gap(higher, lower, bootstrap_repetitions, rng)
            permutation_p = group_permutation_p(higher, lower, group_permutations, rng)

            rows.append(
                {
                    "rule": rule_name,
                    "muscle": muscle,
                    "measure": measure,
                    "n_total": int(len(generated)),
                    "n_valid": int(valid.sum()),
                    "ssr_percent": float(valid.mean() * 100.0),
                    "atlas_reference": atlas[metric],
                    "lower_group_n": int(lower.size),
                    "higher_group_n": int(higher.size),
                    "lower_reference_threshold": lower_threshold,
                    "higher_reference_threshold": higher_threshold,
                    "higher_minus_lower_gap_pp": gap,
                    "gap_bootstrap_ci95_low_pp": ci_low,
                    "gap_bootstrap_ci95_high_pp": ci_high,
                    "welch_p_raw": welch_p,
                    "group_permutation_p_raw": permutation_p,
                    "hedges_g": hedges_g(higher, lower),
                    "bootstrap_repetitions": bootstrap_repetitions,
                    "group_permutations": group_permutations,
                    "analysis_seed": seed,
                }
            )
            counter += 1

    result = pd.DataFrame(rows)
    for measure, index in result.groupby("measure", sort=False).groups.items():
        result.loc[index, "welch_bh_q"] = bh_adjust(result.loc[index, "welch_p_raw"])
        result.loc[index, "group_permutation_bh_q"] = bh_adjust(
            result.loc[index, "group_permutation_p_raw"]
        )
    return result


def make_derangements(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    identity = np.arange(n)
    result = np.empty((count, n), dtype=np.int32)
    accepted = 0
    while accepted < count:
        candidate = rng.permutation(n)
        if np.all(candidate != identity):
            result[accepted] = candidate
            accepted += 1
    return result


def random_pairing_analysis(
    rule_name: str,
    generated: pd.DataFrame,
    gt: pd.DataFrame,
    atlas: dict[str, float],
    observed_metrics: pd.DataFrame,
    permutations: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for measure in MEASURES:
        for muscle in MUSCLES:
            metric = f"{muscle}_{measure}"
            gen = generated[metric].to_numpy(dtype=float)
            gt_values = gt[f"gt_{metric}"].to_numpy(dtype=float)
            delta_matrix = (gen[permutations] - atlas[metric]) / atlas[metric] * 100.0
            null_gap = np.full(permutations.shape[0], np.nan, dtype=float)
            null_n_valid = np.zeros(permutations.shape[0], dtype=np.int16)

            for permutation_index in range(permutations.shape[0]):
                row_delta = delta_matrix[permutation_index]
                valid = np.isfinite(row_delta) & np.isfinite(gt_values)
                row_gt = gt_values[valid]
                row_delta = row_delta[valid]
                null_n_valid[permutation_index] = int(valid.sum())
                if row_gt.size < 5:
                    continue
                lower_threshold = float(np.quantile(row_gt, 0.30))
                higher_threshold = float(np.quantile(row_gt, 0.70))
                lower = row_delta[row_gt <= lower_threshold]
                higher = row_delta[row_gt >= higher_threshold]
                if lower.size and higher.size:
                    null_gap[permutation_index] = higher.mean() - lower.mean()

            observed_row = observed_metrics[
                (observed_metrics["muscle"] == muscle)
                & (observed_metrics["measure"] == measure)
            ].iloc[0]
            observed_gap = float(observed_row["higher_minus_lower_gap_pp"])
            finite_null = null_gap[np.isfinite(null_gap)]
            empirical_p = float(
                (np.sum(np.abs(finite_null) >= abs(observed_gap)) + 1.0)
                / (finite_null.size + 1.0)
            )
            rows.append(
                {
                    "rule": rule_name,
                    "muscle": muscle,
                    "measure": measure,
                    "n_total": int(len(generated)),
                    "matched_n_valid": int(observed_row["n_valid"]),
                    "matched_ssr_percent": float(observed_row["ssr_percent"]),
                    "matched_gap_pp": observed_gap,
                    "random_pairing_gap_median_pp": float(np.median(finite_null)),
                    "random_pairing_gap_95pct_low_pp": float(np.percentile(finite_null, 2.5)),
                    "random_pairing_gap_95pct_high_pp": float(np.percentile(finite_null, 97.5)),
                    "patient_pairing_empirical_p_raw": empirical_p,
                    "valid_random_pairings": int(finite_null.size),
                    "random_pairing_n_valid_min": int(null_n_valid.min()),
                    "random_pairing_n_valid_median": float(np.median(null_n_valid)),
                    "random_pairing_n_valid_max": int(null_n_valid.max()),
                    "pairing_permutations": int(permutations.shape[0]),
                    "pairing_seed": seed,
                    "common_permutation_across_all_metrics": True,
                    "derangement_no_self_matches": True,
                }
            )

    result = pd.DataFrame(rows)
    for measure, index in result.groupby("measure", sort=False).groups.items():
        result.loc[index, "patient_pairing_bh_q"] = bh_adjust(
            result.loc[index, "patient_pairing_empirical_p_raw"]
        )
    return result


def compare_rules(rule_3of3: pd.DataFrame, rule_2of3: pd.DataFrame) -> pd.DataFrame:
    keys = ["muscle", "measure"]
    columns = [
        "n_valid",
        "ssr_percent",
        "higher_minus_lower_gap_pp",
        "gap_bootstrap_ci95_low_pp",
        "gap_bootstrap_ci95_high_pp",
        "welch_p_raw",
        "welch_bh_q",
        "group_permutation_p_raw",
        "group_permutation_bh_q",
        "hedges_g",
    ]
    left = rule_3of3[keys + columns].rename(columns={column: f"rule_3of3_{column}" for column in columns})
    right = rule_2of3[keys + columns].rename(
        columns={column: f"rule_atleast2of3_{column}" for column in columns}
    )
    result = left.merge(right, on=keys, validate="one_to_one")
    result["gap_same_direction"] = (
        np.sign(result["rule_3of3_higher_minus_lower_gap_pp"])
        == np.sign(result["rule_atleast2of3_higher_minus_lower_gap_pp"])
    )
    result["gap_difference_atleast2_minus_3of3_pp"] = (
        result["rule_atleast2of3_higher_minus_lower_gap_pp"]
        - result["rule_3of3_higher_minus_lower_gap_pp"]
    )
    result["raw_welch_significance_reversal"] = (
        (result["rule_3of3_welch_p_raw"] < 0.05)
        != (result["rule_atleast2of3_welch_p_raw"] < 0.05)
    )
    result["bh_welch_significance_reversal"] = (
        (result["rule_3of3_welch_bh_q"] < 0.05)
        != (result["rule_atleast2of3_welch_bh_q"] < 0.05)
    )
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "seed_2026": Path(args.seed_2026_csv),
        "seed_42": Path(args.seed_42_csv),
        "seed_1024": Path(args.seed_1024_csv),
        "seedmean_merged": Path(args.seedmean_merged_csv),
        "atlas": Path(args.atlas_csv),
    }
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    gt, source_rule_3of3 = read_seedmean_merged(input_paths["seedmean_merged"])
    patient_ids = set(gt.index)
    if len(patient_ids) != 74:
        raise ValueError(f"Expected 74 unique patients, found {len(patient_ids)}")
    seeds = {
        2026: read_seed_metrics(input_paths["seed_2026"], patient_ids),
        42: read_seed_metrics(input_paths["seed_42"], patient_ids),
        1024: read_seed_metrics(input_paths["seed_1024"], patient_ids),
    }
    atlas = read_atlas(input_paths["atlas"])

    recomputed_rule_3of3, rule_2of3, audit, patient_audit = build_rule_frames(seeds, gt)
    source_values = source_rule_3of3[metric_names()].to_numpy(dtype=float)
    recomputed_values = recomputed_rule_3of3[metric_names()].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(source_values), np.isnan(recomputed_values)):
        raise ValueError("Existing seedmean missingness pattern differs from recomputed 3/3 rule")
    finite = np.isfinite(source_values) & np.isfinite(recomputed_values)
    seedmean_max_abs_diff = float(
        np.max(np.abs(source_values[finite] - recomputed_values[finite]))
    )
    if seedmean_max_abs_diff > 1e-10:
        raise ValueError(
            f"Existing seedmean values differ from recomputed 3/3 means: {seedmean_max_abs_diff}"
        )
    rule_3of3 = source_rule_3of3[metric_names()].copy()
    rule_3of3.index.name = "file_id"
    rule_3of3.to_csv(output_dir / "rule_3of3_patient_seedmean_metrics.csv")
    rule_2of3.to_csv(output_dir / "rule_atleast2of3_patient_seedmean_metrics.csv")
    audit.to_csv(output_dir / "patient_seed_audit_long.csv", index=False)
    patient_audit.to_csv(output_dir / "patient_seed_audit_by_patient.csv", index=False)

    metrics_3of3 = analyze_rule(
        "3of3_complete_seed_mean",
        rule_3of3,
        gt,
        atlas,
        args.bootstrap,
        args.group_permutations,
        args.seed,
    )
    metrics_2of3 = analyze_rule(
        "atleast2of3_available_seed_mean",
        rule_2of3,
        gt,
        atlas,
        args.bootstrap,
        args.group_permutations,
        args.seed,
    )
    metrics_3of3.to_csv(output_dir / "rule_3of3_full_16_metrics.csv", index=False)
    metrics_2of3.to_csv(output_dir / "rule_atleast2of3_full_16_metrics.csv", index=False)
    comparison = compare_rules(metrics_3of3, metrics_2of3)
    comparison.to_csv(output_dir / "missingness_rule_side_by_side.csv", index=False)

    pairing_rng = np.random.default_rng(args.seed)
    derangements = make_derangements(len(gt), args.pairing_permutations, pairing_rng)
    sorted_ids = gt.index.to_numpy()
    pd.DataFrame(
        {
            "recipient_id": sorted_ids,
            "donor_id": sorted_ids[derangements[0]],
        }
    ).to_csv(output_dir / "first_derangement_mapping_seed2026.csv", index=False)

    pairing_3of3 = random_pairing_analysis(
        "3of3_complete_seed_mean",
        rule_3of3,
        gt,
        atlas,
        metrics_3of3,
        derangements,
        args.seed,
    )
    pairing_2of3 = random_pairing_analysis(
        "atleast2of3_available_seed_mean",
        rule_2of3,
        gt,
        atlas,
        metrics_2of3,
        derangements,
        args.seed,
    )
    pairing_3of3.to_csv(output_dir / "rule_3of3_patient_mispairing_10000.csv", index=False)
    pairing_2of3.to_csv(output_dir / "rule_atleast2of3_patient_mispairing_10000.csv", index=False)

    srr_audit = audit[
        (audit["metric"] == "superior_rectus_right_mcsa")
        & (audit["valid_seed_count"] < 3)
    ].copy()
    srr_audit.to_csv(output_dir / "srr_mcsa_excluded_from_3of3_audit.csv", index=False)

    quality_rows = [
        {"check": "unique_patient_count", "status": "PASS", "value": len(patient_ids), "expected": 74},
        {"check": "seed_id_sets_match_gt", "status": "PASS", "value": True, "expected": True},
        {"check": "gt_metrics_complete", "status": "PASS", "value": int(gt.isna().sum().sum()), "expected": 0},
        {"check": "zero_imputation_used", "status": "PASS", "value": False, "expected": False},
        {
            "check": "existing_seedmean_matches_recomputed_3of3_max_abs_diff",
            "status": "PASS" if seedmean_max_abs_diff <= 1e-10 else "REVIEW",
            "value": seedmean_max_abs_diff,
            "expected": "<=1e-10",
        },
        {
            "check": "srr_3of3_excluded_count",
            "status": "PASS" if len(srr_audit) == 8 else "REVIEW",
            "value": len(srr_audit),
            "expected": 8,
        },
        {
            "check": "srr_excluded_with_exactly_2_valid_seeds",
            "status": "PASS" if int((srr_audit["valid_seed_count"] == 2).sum()) == len(srr_audit) else "REVIEW",
            "value": int((srr_audit["valid_seed_count"] == 2).sum()),
            "expected": len(srr_audit),
        },
        {
            "check": "srr_excluded_in_full74_higher_reference_group",
            "status": "INFO",
            "value": int((srr_audit["full_74_reference_group"] == "higher").sum()),
            "expected": "descriptive",
        },
    ]
    pd.DataFrame(quality_rows).to_csv(output_dir / "data_quality_checks.csv", index=False)

    parameters = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_seed": args.seed,
        "bootstrap_repetitions": args.bootstrap,
        "group_label_permutations": args.group_permutations,
        "patient_pairing_derangements": args.pairing_permutations,
        "patient_count": len(patient_ids),
        "main_missingness_rule": "3/3 seeds valid, then patient-level mean",
        "sensitivity_missingness_rule": ">=2/3 seeds valid, mean available seeds; 1/3 remains missing; no zero imputation",
        "reference_group_rule": "30th/70th percentiles of reference CT values within the metric-specific valid generated-measurement subset",
        "patient_pairing_rule": "one common 74-patient derangement per replicate across all 8 muscles and both measures",
        "bh_families": "separate within 8 MCSA and 8 Volume; separate for each p-value family",
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in input_paths.items()
        },
    }
    (output_dir / "run_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    def select_srr(frame: pd.DataFrame) -> dict[str, object]:
        row = frame[
            (frame["muscle"] == "superior_rectus_right")
            & (frame["measure"] == "mcsa")
        ].iloc[0]
        return row.to_dict()

    summary = {
        "rule_3of3_srr_mcsa": select_srr(metrics_3of3),
        "rule_atleast2of3_srr_mcsa": select_srr(metrics_2of3),
        "rule_3of3_srr_patient_mispairing": select_srr(pairing_3of3),
        "rule_atleast2of3_srr_patient_mispairing": select_srr(pairing_2of3),
        "srr_excluded_count": int(len(srr_audit)),
        "srr_excluded_full74_higher_reference_count": int(
            (srr_audit["full_74_reference_group"] == "higher").sum()
        ),
    }
    (output_dir / "key_results.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, allow_nan=False), encoding="utf-8"
    )

    print(f"[Done] patients: {len(patient_ids)}")
    print(f"[Done] output: {output_dir}")
    print("[SRR 3/3]")
    print(pd.Series(summary["rule_3of3_srr_mcsa"]).to_string())
    print("[SRR >=2/3]")
    print(pd.Series(summary["rule_atleast2of3_srr_mcsa"]).to_string())
    print("[SRR patient mispairing 3/3]")
    print(pd.Series(summary["rule_3of3_srr_patient_mispairing"]).to_string())
    print("[SRR patient mispairing >=2/3]")
    print(pd.Series(summary["rule_atleast2of3_srr_patient_mispairing"]).to_string())


if __name__ == "__main__":
    main()
