from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_rel, wilcoxon

DEFAULT_RESULTS_PATH = Path("results/repeated_active_learning_metrics.csv")

DEFAULT_OUTPUT_DIRECTORY = Path("results/statistics")

DEFAULT_FINAL_SAMPLE_BUDGET = 200

STRATEGY_ORDER = [
    "geographic",
    "hybrid",
    "random",
    "uncertainty",
]

STRATEGY_LABELS = {
    "geographic": "Geographic",
    "hybrid": "Hybrid",
    "random": "Random",
    "uncertainty": "Uncertainty",
}

METRIC_DIRECTION = {
    "Log MAE": "lower",
    "Log RMSE": "lower",
    "Log R2": "higher",
    "Original MAE": "lower",
    "Original RMSE": "lower",
    "Original R2": "higher",
}


def load_repeated_results(
    file_path: str | Path = DEFAULT_RESULTS_PATH,
) -> pd.DataFrame:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Repeated active-learning results not found: {file_path}"
        )

    results = pd.read_csv(file_path)

    required_columns = [
        "Strategy",
        "Random State",
        "Samples Used",
        *METRIC_DIRECTION.keys(),
    ]

    missing_columns = [
        column for column in required_columns if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Repeated results are missing required columns: {missing_columns}"
        )

    results = results.copy()

    results["Strategy"] = results["Strategy"].astype(str).str.strip().str.lower()

    numeric_columns = [
        "Random State",
        "Samples Used",
        *METRIC_DIRECTION.keys(),
    ]

    for column in numeric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="coerce",
        )

    results = results.dropna(
        subset=[
            "Strategy",
            "Random State",
            "Samples Used",
        ]
    )

    duplicated_rows = results.duplicated(
        subset=[
            "Strategy",
            "Random State",
            "Samples Used",
        ],
        keep=False,
    )

    if duplicated_rows.any():
        duplicated_values = results.loc[
            duplicated_rows,
            [
                "Strategy",
                "Random State",
                "Samples Used",
            ],
        ]

        raise ValueError(
            "Each strategy, seed, and sample budget must have only "
            "one result. Duplicate rows were found:\n"
            f"{duplicated_values.to_string(index=False)}"
        )

    return results.sort_values(
        by=[
            "Samples Used",
            "Random State",
            "Strategy",
        ]
    ).reset_index(drop=True)


def prepare_output_directory(
    output_directory: str | Path,
) -> Path:
    """Create the statistical-results directory."""

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


def align_paired_results(
    results: pd.DataFrame,
    strategy_a: str,
    strategy_b: str,
    metric: str,
    sample_budget: int,
) -> pd.DataFrame:
    """
    Align two strategies by random seed for a paired comparison.

    Pairing by random seed is essential because all strategies begin with
    the same initial sample for a given seed.
    """

    if metric not in METRIC_DIRECTION:
        raise ValueError(f"Unknown metric: {metric}")

    budget_results = results[results["Samples Used"] == sample_budget]

    strategy_a_results = budget_results[budget_results["Strategy"] == strategy_a][
        [
            "Random State",
            metric,
        ]
    ].rename(columns={metric: "Strategy A Value"})

    strategy_b_results = budget_results[budget_results["Strategy"] == strategy_b][
        [
            "Random State",
            metric,
        ]
    ].rename(columns={metric: "Strategy B Value"})

    paired_results = strategy_a_results.merge(
        strategy_b_results,
        on="Random State",
        how="inner",
        validate="one_to_one",
    )

    paired_results = paired_results.dropna(
        subset=[
            "Strategy A Value",
            "Strategy B Value",
        ]
    )

    if len(paired_results) < 2:
        raise ValueError(
            "At least two paired runs are required for a "
            f"comparison of {strategy_a} and {strategy_b} "
            f"at budget {sample_budget}."
        )

    return paired_results.sort_values("Random State").reset_index(drop=True)


def calculate_cohens_dz(
    differences: np.ndarray,
) -> float:
    """
    Calculate Cohen's dz for paired observations.

    Cohen's dz equals the mean paired difference divided by the standard
    deviation of the paired differences.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    if len(differences) < 2:
        return np.nan

    difference_standard_deviation = np.std(
        differences,
        ddof=1,
    )

    if np.isclose(
        difference_standard_deviation,
        0,
    ):
        if np.isclose(
            differences.mean(),
            0,
        ):
            return 0.0

        return np.inf if differences.mean() > 0 else -np.inf

    return differences.mean() / difference_standard_deviation


def calculate_rank_biserial_correlation(
    differences: np.ndarray,
) -> float:
    """
    Calculate matched-pairs rank-biserial correlation.

    Positive values mean the paired differences tend to be positive.
    Negative values mean they tend to be negative.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    nonzero_differences = differences[~np.isclose(differences, 0)]

    if len(nonzero_differences) == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero_differences))

    positive_rank_sum = ranks[nonzero_differences > 0].sum()

    negative_rank_sum = ranks[nonzero_differences < 0].sum()

    total_rank_sum = ranks.sum()

    return float((positive_rank_sum - negative_rank_sum) / total_rank_sum)


def classify_effect_size(
    effect_size: float,
) -> str:
    """Classify the absolute magnitude of an effect size."""

    if np.isnan(effect_size):
        return "Unavailable"

    absolute_effect = abs(effect_size)

    if absolute_effect < 0.10:
        return "Negligible"

    if absolute_effect < 0.30:
        return "Small"

    if absolute_effect < 0.50:
        return "Moderate"

    return "Large"


def identify_better_strategy(
    mean_a: float,
    mean_b: float,
    strategy_a: str,
    strategy_b: str,
    metric: str,
) -> str:
    """Identify which strategy has the better mean metric value."""

    direction = METRIC_DIRECTION[metric]

    if np.isclose(
        mean_a,
        mean_b,
    ):
        return "Tie"

    if direction == "lower":
        better_strategy = strategy_a if mean_a < mean_b else strategy_b
    else:
        better_strategy = strategy_a if mean_a > mean_b else strategy_b

    return STRATEGY_LABELS.get(
        better_strategy,
        better_strategy.title(),
    )


def run_pairwise_test(
    results: pd.DataFrame,
    strategy_a: str,
    strategy_b: str,
    metric: str,
    sample_budget: int,
) -> dict[str, float | int | str | bool]:
    """
    Run paired Wilcoxon and paired t-tests for two strategies.

    The Wilcoxon signed-rank test is treated as the primary test because
    only ten repeated runs are available and normality of the paired
    differences should not be assumed.

    The paired t-test is included as a sensitivity analysis.
    """

    paired_results = align_paired_results(
        results=results,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        metric=metric,
        sample_budget=sample_budget,
    )

    values_a = paired_results["Strategy A Value"].to_numpy(dtype=float)

    values_b = paired_results["Strategy B Value"].to_numpy(dtype=float)

    differences = values_a - values_b

    if np.allclose(
        differences,
        0,
    ):
        wilcoxon_statistic = 0.0
        wilcoxon_p_value = 1.0
    else:
        wilcoxon_result = wilcoxon(
            values_a,
            values_b,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        wilcoxon_statistic = float(wilcoxon_result.statistic)

        wilcoxon_p_value = float(wilcoxon_result.pvalue)

    t_test_result = ttest_rel(
        values_a,
        values_b,
        nan_policy="omit",
    )

    rank_biserial = calculate_rank_biserial_correlation(differences)

    cohens_dz = calculate_cohens_dz(differences)

    wins_a = int(
        np.sum(
            values_a < values_b
            if METRIC_DIRECTION[metric] == "lower"
            else values_a > values_b
        )
    )

    wins_b = int(
        np.sum(
            values_b < values_a
            if METRIC_DIRECTION[metric] == "lower"
            else values_b > values_a
        )
    )

    ties = int(
        np.sum(
            np.isclose(
                values_a,
                values_b,
            )
        )
    )

    mean_a = float(np.mean(values_a))

    mean_b = float(np.mean(values_b))

    return {
        "Sample Budget": sample_budget,
        "Metric": metric,
        "Strategy A": STRATEGY_LABELS.get(
            strategy_a,
            strategy_a.title(),
        ),
        "Strategy B": STRATEGY_LABELS.get(
            strategy_b,
            strategy_b.title(),
        ),
        "Number of Pairs": len(paired_results),
        "Strategy A Mean": mean_a,
        "Strategy B Mean": mean_b,
        "Mean Difference A-B": float(np.mean(differences)),
        "Median Difference A-B": float(np.median(differences)),
        "Better Mean Strategy": identify_better_strategy(
            mean_a=mean_a,
            mean_b=mean_b,
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            metric=metric,
        ),
        "Strategy A Wins": wins_a,
        "Strategy B Wins": wins_b,
        "Ties": ties,
        "Wilcoxon Statistic": wilcoxon_statistic,
        "Wilcoxon Raw P": wilcoxon_p_value,
        "Paired T Statistic": float(t_test_result.statistic),
        "Paired T Raw P": float(t_test_result.pvalue),
        "Rank-Biserial Correlation": rank_biserial,
        "Rank-Biserial Magnitude": classify_effect_size(rank_biserial),
        "Cohen Dz": cohens_dz,
        "Cohen Dz Magnitude": classify_effect_size(cohens_dz),
    }


def apply_holm_correction(
    p_values: pd.Series | np.ndarray,
) -> np.ndarray:
    """
    Apply Holm's step-down correction for multiple comparisons.

    Holm correction controls the family-wise error rate and is less
    conservative than a standard Bonferroni correction.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    number_of_tests = len(p_values)

    if number_of_tests == 0:
        return np.array([], dtype=float)

    sorted_indices = np.argsort(p_values)

    sorted_p_values = p_values[sorted_indices]

    adjusted_sorted = np.empty(
        number_of_tests,
        dtype=float,
    )

    running_maximum = 0.0

    for position, p_value in enumerate(sorted_p_values):
        multiplier = number_of_tests - position

        adjusted_value = min(
            p_value * multiplier,
            1.0,
        )

        running_maximum = max(
            running_maximum,
            adjusted_value,
        )

        adjusted_sorted[position] = running_maximum

    adjusted_p_values = np.empty(
        number_of_tests,
        dtype=float,
    )

    adjusted_p_values[sorted_indices] = adjusted_sorted

    return adjusted_p_values


def add_multiple_comparison_corrections(
    test_results: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply Holm correction within each sample-budget and metric family.
    """

    corrected_groups = []

    for _, group in test_results.groupby(
        [
            "Sample Budget",
            "Metric",
        ],
        sort=False,
    ):
        corrected_group = group.copy()

        corrected_group["Wilcoxon Holm P"] = apply_holm_correction(
            corrected_group["Wilcoxon Raw P"]
        )

        corrected_group["Paired T Holm P"] = apply_holm_correction(
            corrected_group["Paired T Raw P"]
        )

        corrected_group["Wilcoxon Significant"] = (
            corrected_group["Wilcoxon Holm P"] < alpha
        )

        corrected_group["Paired T Significant"] = (
            corrected_group["Paired T Holm P"] < alpha
        )

        corrected_groups.append(corrected_group)

    return pd.concat(
        corrected_groups,
        ignore_index=True,
    )


def run_all_pairwise_tests(
    results: pd.DataFrame,
    metrics: list[str] | None = None,
    sample_budgets: list[int] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compare every pair of strategies at selected sampling budgets.
    """

    if metrics is None:
        metrics = [
            "Log RMSE",
            "Log MAE",
            "Log R2",
        ]

    invalid_metrics = [metric for metric in metrics if metric not in METRIC_DIRECTION]

    if invalid_metrics:
        raise ValueError(f"Unknown metrics requested: {invalid_metrics}")

    if sample_budgets is None:
        sample_budgets = sorted(
            results["Samples Used"].dropna().astype(int).unique().tolist()
        )

    strategy_pairs = list(
        combinations(
            STRATEGY_ORDER,
            2,
        )
    )

    all_tests = []

    for sample_budget in sample_budgets:
        for metric in metrics:
            for strategy_a, strategy_b in strategy_pairs:
                test_result = run_pairwise_test(
                    results=results,
                    strategy_a=strategy_a,
                    strategy_b=strategy_b,
                    metric=metric,
                    sample_budget=sample_budget,
                )

                all_tests.append(test_result)

    raw_results = pd.DataFrame(all_tests)

    corrected_results = add_multiple_comparison_corrections(
        test_results=raw_results,
        alpha=alpha,
    )

    return corrected_results.sort_values(
        by=[
            "Sample Budget",
            "Metric",
            "Wilcoxon Holm P",
        ]
    ).reset_index(drop=True)


def extract_primary_comparisons(
    statistical_results: pd.DataFrame,
    sample_budget: int,
    metric: str = "Log RMSE",
) -> pd.DataFrame:
    """
    Extract the most important comparisons for the paper.

    The hybrid strategy is compared with geographic, random, and
    uncertainty sampling.
    """

    primary_pairs = {
        frozenset(
            [
                "Hybrid",
                "Geographic",
            ]
        ),
        frozenset(
            [
                "Hybrid",
                "Random",
            ]
        ),
        frozenset(
            [
                "Hybrid",
                "Uncertainty",
            ]
        ),
    }
    filtered_results = statistical_results[
        (statistical_results["Sample Budget"] == sample_budget)
        & (statistical_results["Metric"] == metric)
    ].copy()

    pair_is_primary = filtered_results.apply(
        lambda row: (
            frozenset(
                [
                    row["Strategy A"],
                    row["Strategy B"],
                ]
            )
            in primary_pairs
        ),
        axis=1,
    )

    return (
        filtered_results[pair_is_primary]
        .sort_values("Wilcoxon Holm P")
        .reset_index(drop=True)
    )


def summarize_significant_results(
    statistical_results: pd.DataFrame,
    metric: str = "Log RMSE",
) -> pd.DataFrame:
    """Return statistically significant Wilcoxon comparisons."""

    return (
        statistical_results[
            (statistical_results["Metric"] == metric)
            & (statistical_results["Wilcoxon Significant"])
        ]
        .sort_values(
            by=[
                "Sample Budget",
                "Wilcoxon Holm P",
            ]
        )
        .reset_index(drop=True)
    )


def run_statistical_analysis(
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    final_sample_budget: int = DEFAULT_FINAL_SAMPLE_BUDGET,
    alpha: float = 0.05,
) -> None:
    """
    Run and save the complete statistical analysis.

    The primary outcome is log RMSE. Log MAE and log R² are included as
    secondary metrics.
    """

    results = load_repeated_results(results_path)

    output_directory = prepare_output_directory(output_directory)

    available_budgets = sorted(results["Samples Used"].astype(int).unique().tolist())

    all_budget_results = run_all_pairwise_tests(
        results=results,
        metrics=[
            "Log RMSE",
            "Log MAE",
            "Log R2",
        ],
        sample_budgets=available_budgets,
        alpha=alpha,
    )

    final_budget_results = all_budget_results[
        all_budget_results["Sample Budget"] == final_sample_budget
    ].reset_index(drop=True)

    primary_results = extract_primary_comparisons(
        statistical_results=all_budget_results,
        sample_budget=final_sample_budget,
        metric="Log RMSE",
    )

    significant_log_rmse_results = summarize_significant_results(
        statistical_results=all_budget_results,
        metric="Log RMSE",
    )

    all_budget_results.to_csv(
        output_directory / "all_pairwise_tests.csv",
        index=False,
    )

    final_budget_results.to_csv(
        output_directory / "final_budget_tests.csv",
        index=False,
    )

    primary_results.to_csv(
        output_directory / "primary_log_rmse_comparisons.csv",
        index=False,
    )

    significant_log_rmse_results.to_csv(
        output_directory / "significant_log_rmse_results.csv",
        index=False,
    )

    print("\nStatistical analysis complete.")

    print(f"\nPrimary Log RMSE comparisons at {final_sample_budget} samples:")

    display_columns = [
        "Strategy A",
        "Strategy B",
        "Strategy A Mean",
        "Strategy B Mean",
        "Better Mean Strategy",
        "Wilcoxon Holm P",
        "Wilcoxon Significant",
        "Rank-Biserial Correlation",
        "Rank-Biserial Magnitude",
    ]

    print(primary_results[display_columns].to_string(index=False))

    print(f"\nStatistical results saved to: {output_directory}")


if __name__ == "__main__":
    run_statistical_analysis()
