#!/usr/bin/env python3
from __future__ import annotations

from functools import reduce
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

# ---- Pretty-name mappings (kept identical semantics) ---------------------------------

_RISK_MAP = {"BayesRisk": "R_b", "ExcessRisk": "R_e", "TotalRisk": "R_t"}
_APP_MAP = {"outer": "1", "inner": "2", "central": "3"}
_SCORE_MAP = {
    "LogScore": "Logscore",
    "BrierScore": "Brier",
    "SphericalScore": "Spherical",
    "ZeroOneScore": "Zero-one",
}

# Ordering rules used by sort_rows (unchanged)
_IND_ORDER = {"cifar10": 0, "cifar100": 1, "tiny_imagenet": 2}
_BASE_ORDER = {
    "cifar10": 0,
    "cifar100": 1,
    "svhn": 2,
    "tiny_imagenet": 3,
    "imagenet_a": 4,
    "imagenet_r": 5,
    "imagenet_o": 6,
}
_SUFFIX_ORDER = {"[miscls]": 0, "[ood]": 1, "[selective]": 2}

# Evaluation datasets order (unchanged)
_EVAL_ORDER: Sequence[str] = (
    "cifar10",
    "cifar100",
    "svhn",
    "tiny_imagenet",
    "imagenet_a",
    "imagenet_r",
    "imagenet_o",
)


# ---- Core helpers ---------------------------------------------------------------------


def prettify_measure(name: str) -> str:
    """
    Map internal 'measure' names to the display labels used in tables.
    Behavior preserved exactly (including case and spacing).
    """
    n = str(name)
    low = n.lower()

    if low == "gmm":
        return "gmm"
    if low == "mahalanobis":
        return "mahalanobis"

    if n.startswith("Risk_"):
        parts = n.split("_")
        score = parts[1] if len(parts) > 1 else "Score"
        risk_type = parts[2] if len(parts) > 2 else "Risk"
        gt_app = parts[3] if len(parts) > 3 else ""
        pred_app = parts[4] if len(parts) > 4 else ""

        r = _RISK_MAP.get(risk_type, risk_type)
        a = _APP_MAP.get(gt_app, gt_app)
        b = _APP_MAP.get(pred_app, pred_app)
        s = _SCORE_MAP.get(score, score)

        if risk_type == "BayesRisk":
            return f"{r} {a} ({s})".strip()
        return f"{r} {a} {b} ({s})".strip()

    return low


def single_row_from(df_subset: pd.DataFrame, value_col: str, row_label) -> pd.DataFrame:
    """
    Aggregate a subset into a single-row dataframe indexed by `row_label`,
    taking the mean per 'measure'.
    """
    if df_subset.empty:
        return pd.DataFrame()
    s = df_subset.groupby("measure", sort=False)[value_col].mean()
    row = s.to_frame().T
    row.index = [row_label]
    return row


def build_rows_for_ind(
    df_sub: pd.DataFrame, ind: str, selective_metric: str = "acc_cov_auc"
) -> pd.DataFrame:
    """
    Build per-evaluation rows for a fixed in-distribution dataset `ind`.
    Preserves original behavior and exceptions.
    """
    frames: List[pd.DataFrame] = []

    for eval_ds in _EVAL_ORDER:
        if eval_ds == ind:
            # Misclassification detection (ROC AUC)
            mask_mis = (df_sub["problem_type"] == "misclassification_detection") & (
                df_sub["ood_dataset"] == eval_ds
            )
            frames.append(
                single_row_from(
                    df_sub[mask_mis], "roc_auc", (ind, f"{eval_ds} [miscls]")
                )
            )

            # Selective prediction (area-like metric)
            if selective_metric not in df_sub.columns:
                raise KeyError(
                    f"Selective metric '{selective_metric}' not found in columns: {list(df_sub.columns)}"
                )
            mask_sel = (df_sub["problem_type"] == "selective_prediction") & (
                df_sub["ood_dataset"] == eval_ds
            )
            frames.append(
                single_row_from(
                    df_sub[mask_sel], selective_metric, (ind, f"{eval_ds} [selective]")
                )
            )
        else:
            # OOD detection (ROC AUC)
            mask_ood = (df_sub["problem_type"] == "ood_detection") & (
                df_sub["ood_dataset"] == eval_ds
            )
            frames.append(
                single_row_from(df_sub[mask_ood], "roc_auc", (ind, f"{eval_ds} [ood]"))
            )

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=0, sort=True)
    out.index = pd.MultiIndex.from_tuples(out.index, names=["ind_dataset", "eval"])
    return out


def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stable sorting of rows according to ind dataset, base eval name, and suffix.
    Output identical to original function.
    """
    order_df = df.reset_index()

    def base_name(eval_label: str) -> str:
        return eval_label.split()[0]

    def suffix(eval_label: str) -> str:
        return eval_label[eval_label.find("[") :] if "[" in eval_label else ""

    order_df["_ord1"] = order_df["ind_dataset"].map(_IND_ORDER).fillna(99)
    order_df["_base"] = order_df["eval"].map(base_name)
    order_df["_ord2"] = order_df["_base"].map(_BASE_ORDER).fillna(99)
    order_df["_suf"] = order_df["eval"].map(suffix)
    order_df["_ord3"] = order_df["_suf"].map(_SUFFIX_ORDER).fillna(99)

    order_df = order_df.sort_values(by=["_ord1", "_ord2", "_ord3"]).drop(
        columns=["_ord1", "_ord2", "_ord3", "_base", "_suf"]
    )
    return order_df.set_index(["ind_dataset", "eval"])


def transform_by_tasks(
    df: pd.DataFrame, selective_metric: str = "acc_cov_auc", include_std: bool = False
):
    """
    Transform the raw dataframe into a readable table (mean over ensemble groups).
    Returns a single DataFrame or (mean_table, std_table) if include_std is True.
    Preserves exceptions and averaging logic exactly.
    """
    required = {
        "ind_dataset",
        "ood_dataset",
        "measure",
        "problem_type",
        "ensemble_group",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Input CSV is missing required columns: {sorted(missing)}")

    # Prettify measure names
    df = df.copy()
    df["measure"] = df["measure"].apply(prettify_measure)

    # Average over ensemble groups: build per-group tables then average
    groups = (
        pd.to_numeric(df["ensemble_group"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    groups = sorted(groups.tolist())

    tables: List[pd.DataFrame] = []
    for g in groups:
        df_g = df[df["ensemble_group"].astype(str) == str(g)]
        parts = [
            build_rows_for_ind(df_g[df_g["ind_dataset"] == ind], ind, selective_metric)
            for ind in ("cifar10", "cifar100", "tiny_imagenet")
        ]
        tbl = pd.concat(parts, axis=0, sort=True)
        tables.append(tbl)

    if not tables:
        raise RuntimeError(
            "No tables could be built from the provided data. Check filters and columns."
        )

    # Align all tables to the union of index/columns (preserves NaNs just like original)
    all_index = sorted(set().union(*(set(t.index) for t in tables)))
    all_columns = sorted(set().union(*(set(t.columns) for t in tables)))
    aligned = [t.reindex(index=all_index, columns=all_columns) for t in tables]

    # Mean across groups: keep identical behavior (sum with fill_value=0 then / len)
    avg_table = reduce(lambda a, b: a.add(b, fill_value=0), aligned) / len(aligned)
    avg_table = sort_rows(avg_table)

    if not include_std:
        return avg_table

    # Standard deviation across groups (unchanged: NaNs propagate)
    import numpy as np  # local import kept

    stacked = np.stack([t.values for t in aligned], axis=0)
    std_values = np.std(stacked, axis=0, ddof=1 if len(aligned) > 1 else 0)
    std_table = pd.DataFrame(
        std_values, index=pd.MultiIndex.from_tuples(all_index), columns=all_columns
    )
    std_table.index.names = ["ind_dataset", "eval"]
    std_table = sort_rows(std_table)
    return avg_table, std_table


# ---- Composites -----------------------------------------------------------------------


def select_composite_and_components(
    transformed_df: pd.DataFrame, composite_name: str
) -> pd.DataFrame:
    """
    Select a composite measure and its component 1D measures from transformed_df.
    Behavior (errors, prints, matching) preserved.
    """
    from configs.interesting_compositions import INTERESTING_COMPOSITIONS  # noqa

    def _convert_config_name_to_prettified(config_name: str) -> str:
        """
        Convert config print_name to prettified DataFrame column.
        "B 1 (log)" -> "R_b 1 (Logscore)"
        "EXC 1 1 (brier)" -> "R_e 1 1 (Brier)"
        """
        low = config_name.lower()
        if low == "mahalanobis score":
            return "mahalanobis"
        if low == "gmm score":
            return "gmm"

        import re

        m = re.match(r"(\w+)\s+(\d+)(?:\s+(\d+))?\s+\(([^)]+)\)", config_name)
        if not m:
            return low  # fallback identical to original

        risk_part, gt_app, pred_approx, score_part = m.groups()
        risk_pretty = {"B": "R_b", "EXC": "R_e", "TOT": "R_t"}.get(risk_part, risk_part)
        score_pretty = {
            "log": "Logscore",
            "brier": "Brier",
            "sph": "Spherical",
            "zero one": "Zero-one",
        }.get(score_part, score_part)

        if pred_approx:
            return f"{risk_pretty} {gt_app} {pred_approx} ({score_pretty})"
        return f"{risk_pretty} {gt_app} ({score_pretty})"

    if composite_name not in INTERESTING_COMPOSITIONS:
        available = list(INTERESTING_COMPOSITIONS.keys())
        raise KeyError(
            f"Composite measure '{composite_name}' not found in INTERESTING_COMPOSITIONS. "
            f"Available compositions: {available}"
        )

    component_configs = INTERESTING_COMPOSITIONS[composite_name]

    component_names: List[str] = []
    prettified_names: List[str] = []
    for config in component_configs:
        if isinstance(config, dict) and "print_name" in config:
            config_name = config["print_name"]
            prettified_name = _convert_config_name_to_prettified(config_name)
            component_names.append(config_name)
            prettified_names.append(prettified_name)
        else:
            print(f"Warning: Skipping invalid config: {config}")

    if not prettified_names:
        raise ValueError(
            f"No valid component names found for composite '{composite_name}'"
        )

    available_columns = list(transformed_df.columns)
    matching_columns: List[str] = []
    name_mapping: Dict[str, str] = {}

    for config_name, pretty in zip(component_names, prettified_names):
        if pretty in available_columns:
            matching_columns.append(pretty)
            name_mapping[pretty] = config_name

    if not matching_columns:
        print(f"Config component names: {component_names}")
        print(f"Prettified component names: {prettified_names}")
        print(f"Available DataFrame columns: {available_columns}")
        raise ValueError(
            f"None of the component measures for '{composite_name}' were found in the DataFrame columns"
        )

    result_df = transformed_df[matching_columns].copy()

    composite_pretty = composite_name.lower()
    if composite_pretty in available_columns:
        result_df[composite_pretty] = transformed_df[composite_pretty]
        matching_columns.append(composite_pretty)
    else:
        print(
            f"Note: Composite measure '{composite_pretty}' not found in DataFrame columns"
        )

    if len(matching_columns) - (1 if composite_pretty in matching_columns else 0) < len(
        prettified_names
    ):
        missing_prettified = [n for n in prettified_names if n not in available_columns]
        missing_config = [
            component_names[prettified_names.index(n)] for n in missing_prettified
        ]
        print(
            f"Note: {len(missing_prettified)} component measures were not found in DataFrame:"
        )
        for mp, mc in zip(missing_prettified, missing_config):
            print(f"  - {mp} (from config: {mc})")

    return result_df


def check_composite_dominance(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, check whether the 'composite*' column outperforms other measures
    by >=100%, >=75%, and >=50% of them (higher is better).
    Also checks if composite is better than the worst of its components.
    """
    composite_cols = [c for c in df.columns if c.startswith("composite")]
    if not composite_cols:
        raise ValueError("No column starting with 'composite' found in DataFrame")

    composite_col = composite_cols[0]
    other_cols = [c for c in df.columns if not c.startswith("composite")]
    if not other_cols:
        raise ValueError("No non-composite columns found for comparison")

    result_df = df.copy()
    dominance_100: List[bool] = []
    dominance_75: List[bool] = []
    dominance_50: List[bool] = []
    beats_worst: List[bool] = []

    for _, row in df.iterrows():
        comp_val = row[composite_col]
        if pd.isna(comp_val):
            dominance_100.append(False)
            dominance_75.append(False)
            dominance_50.append(False)
            beats_worst.append(False)
            continue

        other_values = [row[c] for c in other_cols if pd.notna(row[c])]
        if not other_values:
            dominance_100.append(False)
            dominance_75.append(False)
            dominance_50.append(False)
            beats_worst.append(False)
            continue

        beats_count = sum(1 for v in other_values if comp_val > v)
        total = len(other_values)
        pct = beats_count / total if total else 0.0

        dominance_100.append(pct >= 1.0)
        dominance_75.append(pct >= 0.75)
        dominance_50.append(pct >= 0.50)

        # Check if composite beats the worst component
        worst_component = min(other_values)
        beats_worst.append(comp_val > worst_component)

    result_df["if_dominates_100%"] = dominance_100
    result_df["if_dominates_75%"] = dominance_75
    result_df["if_dominates_50%"] = dominance_50
    result_df["beats_worst_component"] = beats_worst
    return result_df


# ---- Pareto utilities ----------------------------------------------------------------


def pareto_front(points: Sequence[Tuple[float, float]]) -> List[int]:
    """Return indices of Pareto-optimal points (2D). Behavior unchanged."""
    pareto: List[int] = []
    for i, (x, y) in enumerate(points):
        dominated = any(
            (x2 >= x and y2 >= y) and (x2 > x or y2 > y)
            for j, (x2, y2) in enumerate(points)
            if j != i
        )
        if not dominated:
            pareto.append(i)
    return pareto


def pareto_depth(points: Sequence[Tuple[float, float]]) -> List[int]:
    """
    Calculate Pareto depth for each point in 2D space.

    Returns a list where depth[i] is the Pareto depth of points[i].
    Depth 0 = on Pareto front, depth 1 = dominated only by Pareto front points, etc.
    """
    n = len(points)
    if n == 0:
        return []

    depths = [-1] * n  # -1 means not yet assigned
    remaining_indices = list(range(n))
    current_depth = 0

    while remaining_indices:
        # Find Pareto front among remaining points
        remaining_points = [points[i] for i in remaining_indices]
        front_indices_in_remaining = pareto_front(remaining_points)

        # Map back to original indices and assign current depth
        for idx_in_remaining in front_indices_in_remaining:
            original_idx = remaining_indices[idx_in_remaining]
            depths[original_idx] = current_depth

        # Remove points that were assigned a depth
        remaining_indices = [
            idx
            for i, idx in enumerate(remaining_indices)
            if i not in front_indices_in_remaining
        ]
        current_depth += 1

    return depths


def analyze_composite_pareto_performance(
    transformed_df: pd.DataFrame,
    composite_names: Dict[str, object],
    do_for_each_measure: bool = False,
) -> Dict[str, Dict[str, object]]:
    """
    For each composite, count how often it lies on the Pareto front
    of its components across all pairs of problems. Returns stats dict.

    When do_for_each_measure=True, also calculates Pareto stats for each individual component.
    """
    import itertools
    from typing import Union

    import pandas as pd  # local import preserved

    composite_pareto_results: Dict[
        str, Dict[str, Union[float, Dict[str, Dict[str, float]]]]
    ] = {}

    for composite_name in composite_names.keys():
        try:
            composite_df = select_composite_and_components(
                transformed_df, composite_name
            )

            composite_cols = [
                c for c in composite_df.columns if c.startswith("composite")
            ]
            if not composite_cols:
                print(f"No composite column found for {composite_name}")
                continue

            composite_col = composite_cols[0]
            component_cols = [
                c for c in composite_df.columns if not c.startswith("composite")
            ]
            if len(component_cols) < 2:
                print(f"Not enough components for {composite_name} (need at least 2)")
                continue

            problems = list(composite_df.index)

            # Initialize results for this composite
            result_dict = {}

            # Calculate composite Pareto stats
            composite_pareto_count = 0
            total_pairs = 0
            composite_depths = []  # Track all depth values for composite

            # If analyzing individual measures, initialize counters for each component
            if do_for_each_measure:
                component_pareto_counts = {col: 0 for col in component_cols}
                component_total_pairs = {col: 0 for col in component_cols}
                component_depths = {
                    col: [] for col in component_cols
                }  # Track depths for each component

            for problem1, problem2 in itertools.combinations(problems, 2):
                row1 = composite_df.loc[problem1]
                row2 = composite_df.loc[problem2]

                c1, c2 = row1[composite_col], row2[composite_col]
                if pd.isna(c1) or pd.isna(c2):
                    continue

                points: List[Tuple[float, float]] = []
                valid_component_indices = []

                # components
                for i, col in enumerate(component_cols):
                    v1, v2 = row1[col], row2[col]
                    if pd.notna(v1) and pd.notna(v2):
                        points.append((v1, v2))
                        valid_component_indices.append(i)

                # composite
                points.append((c1, c2))

                # Need >= 3 points (2 comps + 1 composite) to be meaningful
                if len(points) < 3:
                    continue

                pareto_indices = pareto_front(points)
                depths = pareto_depth(points)

                # Check if composite is on Pareto front (it's the last point added)
                composite_idx = len(points) - 1
                composite_depths.append(depths[composite_idx])
                if composite_idx in pareto_indices:
                    composite_pareto_count += 1
                total_pairs += 1

                # If analyzing individual measures, check each component
                if do_for_each_measure:
                    for point_idx in range(
                        len(points) - 1
                    ):  # Exclude composite (last point)
                        component_idx = valid_component_indices[point_idx]
                        col = component_cols[component_idx]

                        component_depths[col].append(depths[point_idx])
                        if point_idx in pareto_indices:
                            component_pareto_counts[col] += 1
                        component_total_pairs[col] += 1

            # Store composite results
            if total_pairs > 0:
                import numpy as np

                composite_pct = (composite_pareto_count / total_pairs) * 100
                composite_avg_depth = (
                    np.mean(composite_depths) if composite_depths else 0.0
                )
                composite_median_depth = (
                    np.median(composite_depths) if composite_depths else 0.0
                )

                result_dict.update(
                    {
                        "pareto_count": composite_pareto_count,
                        "total_problems": total_pairs,
                        "pareto_percentage": composite_pct,
                        "average_pareto_depth": composite_avg_depth,
                        "median_pareto_depth": composite_median_depth,
                    }
                )

                # Store individual component results if requested
                if do_for_each_measure:
                    individual_measures = {}
                    for col in component_cols:
                        if component_total_pairs[col] > 0:
                            component_pct = (
                                component_pareto_counts[col]
                                / component_total_pairs[col]
                            ) * 100
                            component_avg_depth = (
                                np.mean(component_depths[col])
                                if component_depths[col]
                                else 0.0
                            )
                            component_median_depth = (
                                np.median(component_depths[col])
                                if component_depths[col]
                                else 0.0
                            )

                            individual_measures[col] = {
                                "pareto_count": component_pareto_counts[col],
                                "total_problems": component_total_pairs[col],
                                "pareto_percentage": component_pct,
                                "average_pareto_depth": component_avg_depth,
                                "median_pareto_depth": component_median_depth,
                            }

                    result_dict["individual_measures"] = individual_measures

                composite_pareto_results[composite_name] = result_dict

        except Exception as e:
            print(f"Error analyzing {composite_name}: {e}")

    return composite_pareto_results


# ---- Ranking --------------------------------------------------------------------------


def compute_average_ranks(transformed_df: pd.DataFrame) -> pd.Series:
    """
    Compute average ranks (dense) per metric across all (ind_dataset, eval) rows.
    Higher values are better (unchanged).
    """
    import numpy as np
    import pandas as pd  # local imports preserved to match original style
    from scipy.stats import rankdata

    df = transformed_df.reset_index()
    metrics = [c for c in df.columns if c not in ["ind_dataset", "eval"]]

    all_ranks: List[pd.Series] = []
    for _, row in df.iterrows():
        metric_values = row[metrics]
        valid_mask = ~pd.isna(metric_values)
        valid_metrics = metric_values[valid_mask]
        valid_names = [metrics[i] for i in range(len(metrics)) if valid_mask.iloc[i]]
        if len(valid_metrics) == 0:
            continue

        ranks = rankdata(-valid_metrics, method="dense")
        all_ranks.append(pd.Series(ranks, index=valid_names))

    if not all_ranks:
        raise ValueError("No valid data found to compute ranks")

    ranks_df = pd.concat(all_ranks, axis=1).T
    average_ranks = ranks_df.mean(axis=0, skipna=True).sort_values()
    return average_ranks
