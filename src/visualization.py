"""Generate publication-quality figures from saved experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SUMMARY_PATH = Path("results/repeated_active_learning_summary.csv")

DEFAULT_FIGURES_DIRECTORY = Path("figures")

DEFAULT_NUMBER_OF_RUNS = 10

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


def load_active_learning_summary(
    file_path: str | Path = DEFAULT_SUMMARY_PATH,
) -> pd.DataFrame:
    """Load and validate the repeated active-learning summary CSV."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Active-learning summary file not found: {file_path}")

    results = pd.read_csv(file_path)

    required_columns = [
        "Strategy",
        "Samples Used",
        "Log MAE mean",
        "Log MAE std",
        "Log RMSE mean",
        "Log RMSE std",
        "Log R2 mean",
        "Log R2 std",
        "Original MAE mean",
        "Original MAE std",
        "Original RMSE mean",
        "Original RMSE std",
        "Original R2 mean",
        "Original R2 std",
    ]

    missing_columns = [
        column for column in required_columns if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Summary results are missing required columns: {missing_columns}"
        )

    results = results.copy()

    results["Strategy"] = results["Strategy"].astype(str).str.strip().str.lower()

    numeric_columns = [column for column in required_columns if column != "Strategy"]

    for column in numeric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="coerce",
        )

    results = results.dropna(
        subset=[
            "Strategy",
            "Samples Used",
        ]
    )

    available_strategies = set(results["Strategy"].unique())

    missing_strategies = [
        strategy for strategy in STRATEGY_ORDER if strategy not in available_strategies
    ]

    if missing_strategies:
        print(f"Warning: results do not contain these strategies: {missing_strategies}")

    return results.sort_values(
        by=[
            "Strategy",
            "Samples Used",
        ]
    ).reset_index(drop=True)


def prepare_output_directories(
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """
    Create separate directories for primary and supplementary figures.
    """

    output_directory = Path(output_directory)

    primary_directory = output_directory / "primary"

    supplementary_directory = output_directory / "supplementary"

    primary_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    supplementary_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        primary_directory,
        supplementary_directory,
    )


def calculate_confidence_interval(
    standard_deviation: np.ndarray,
    number_of_runs: int,
    confidence_multiplier: float = 1.96,
) -> np.ndarray:
    """
    Calculate the half-width of an approximate 95% confidence interval.

    The formula is:

        1.96 × standard deviation / sqrt(number of runs)
    """

    if number_of_runs <= 1:
        raise ValueError("number_of_runs must be greater than one.")

    standard_deviation = np.asarray(
        standard_deviation,
        dtype=float,
    )

    return confidence_multiplier * standard_deviation / np.sqrt(number_of_runs)


def plot_learning_curve(
    results: pd.DataFrame,
    mean_column: str,
    std_column: str,
    y_axis_label: str,
    title: str,
    output_path: str | Path,
    number_of_runs: int = DEFAULT_NUMBER_OF_RUNS,
    zero_reference_line: bool = False,
) -> None:
    """
    Plot mean performance with approximate 95% confidence intervals.
    """

    output_path = Path(output_path)

    fig, ax = plt.subplots(figsize=(9, 6))

    plotted_strategy = False

    for strategy in STRATEGY_ORDER:
        strategy_results = results[results["Strategy"] == strategy].sort_values(
            "Samples Used"
        )

        if strategy_results.empty:
            continue

        plotted_strategy = True

        samples = strategy_results["Samples Used"].to_numpy(dtype=float)

        means = strategy_results[mean_column].to_numpy(dtype=float)

        standard_deviations = (
            strategy_results[std_column].fillna(0).to_numpy(dtype=float)
        )

        confidence_intervals = calculate_confidence_interval(
            standard_deviation=standard_deviations,
            number_of_runs=number_of_runs,
        )

        line = ax.plot(
            samples,
            means,
            marker="o",
            markersize=4,
            linewidth=2,
            label=STRATEGY_LABELS.get(
                strategy,
                strategy.title(),
            ),
        )[0]

        ax.fill_between(
            samples,
            means - confidence_intervals,
            means + confidence_intervals,
            alpha=0.08,
            color=line.get_color(),
        )

    if not plotted_strategy:
        raise ValueError("No recognized sampling strategies were found.")

    if zero_reference_line:
        ax.axhline(
            y=0,
            linestyle="--",
            linewidth=1,
            alpha=0.8,
        )

    ax.set_title(
        title,
        fontsize=14,
        pad=12,
    )

    ax.set_xlabel(
        "Number of labeled samples",
        fontsize=11,
    )

    ax.set_ylabel(
        y_axis_label,
        fontsize=11,
    )

    ax.grid(
        visible=True,
        alpha=0.20,
    )

    ax.legend(
        title="Sampling strategy",
        frameon=True,
    )

    fig.text(
        0.5,
        0.01,
        (
            "Shaded regions represent approximate "
            "95% confidence intervals across repeated runs."
        ),
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1,
        ]
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_final_strategy_comparison(
    results: pd.DataFrame,
    sample_budget: int,
    output_path: str | Path,
    number_of_runs: int = DEFAULT_NUMBER_OF_RUNS,
) -> None:
    """
    Plot final log RMSE means and 95% confidence intervals.

    A point-and-error-bar plot is used because the strategy differences
    are relatively small and are easier to compare without bars.
    """

    output_path = Path(output_path)

    final_results = results[results["Samples Used"] == sample_budget].copy()

    if final_results.empty:
        raise ValueError(f"No results found for a sample budget of {sample_budget}.")

    order_mapping = {
        strategy: position for position, strategy in enumerate(STRATEGY_ORDER)
    }

    final_results["Strategy Order"] = final_results["Strategy"].map(order_mapping)

    final_results = final_results.dropna(subset=["Strategy Order"]).sort_values(
        "Strategy Order"
    )

    labels = [
        STRATEGY_LABELS.get(
            strategy,
            strategy.title(),
        )
        for strategy in final_results["Strategy"]
    ]

    means = final_results["Log RMSE mean"].to_numpy(dtype=float)

    standard_deviations = final_results["Log RMSE std"].fillna(0).to_numpy(dtype=float)

    confidence_intervals = calculate_confidence_interval(
        standard_deviation=standard_deviations,
        number_of_runs=number_of_runs,
    )

    positions = np.arange(len(final_results))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.errorbar(
        positions,
        means,
        yerr=confidence_intervals,
        fmt="o",
        markersize=8,
        capsize=6,
        linewidth=2,
    )

    ax.set_xticks(
        positions,
        labels,
    )

    ax.set_title(
        (f"Mean Log RMSE at {sample_budget} Labeled Samples"),
        fontsize=14,
        pad=12,
    )

    ax.set_xlabel(
        "Sampling strategy",
        fontsize=11,
    )

    ax.set_ylabel(
        "Mean log RMSE",
        fontsize=11,
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    vertical_offset = max(
        confidence_intervals.max() * 0.25,
        0.002,
    )

    for position, mean, interval in zip(
        positions,
        means,
        confidence_intervals,
        strict=False,
    ):
        ax.text(
            position,
            mean + interval + vertical_offset,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.text(
        0.5,
        0.01,
        (
            "Error bars represent approximate "
            "95% confidence intervals across repeated runs."
        ),
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1,
        ]
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_improvement_over_random(
    results: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot mean log RMSE improvement relative to random sampling.

    Improvement is calculated as:

        random log RMSE - strategy log RMSE

    Positive values mean that the strategy has lower error than random.
    Negative values mean that the strategy has higher error than random.
    """

    output_path = Path(output_path)

    random_results = results[results["Strategy"] == "random"][
        [
            "Samples Used",
            "Log RMSE mean",
        ]
    ].rename(columns={"Log RMSE mean": "Random Log RMSE"})

    if random_results.empty:
        raise ValueError(
            "Random sampling results are required to calculate improvement."
        )

    comparison_results = results[results["Strategy"] != "random"].merge(
        random_results,
        on="Samples Used",
        how="inner",
    )

    comparison_results["Improvement Over Random"] = (
        comparison_results["Random Log RMSE"] - comparison_results["Log RMSE mean"]
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for strategy in [
        "geographic",
        "hybrid",
        "uncertainty",
    ]:
        strategy_results = comparison_results[
            comparison_results["Strategy"] == strategy
        ].sort_values("Samples Used")

        if strategy_results.empty:
            continue

        ax.plot(
            strategy_results["Samples Used"],
            strategy_results["Improvement Over Random"],
            marker="o",
            markersize=4,
            linewidth=2,
            label=STRATEGY_LABELS.get(
                strategy,
                strategy.title(),
            ),
        )

    ax.axhline(
        y=0,
        linewidth=1.2,
        linestyle="--",
        alpha=0.8,
    )

    ax.set_title(
        "Log RMSE Improvement Relative to Random Sampling",
        fontsize=14,
        pad=12,
    )

    ax.set_xlabel(
        "Number of labeled samples",
        fontsize=11,
    )

    ax.set_ylabel(
        "Random log RMSE − strategy log RMSE",
        fontsize=11,
    )

    ax.grid(
        visible=True,
        alpha=0.20,
    )

    ax.legend(
        title="Sampling strategy",
        frameon=True,
    )

    fig.text(
        0.5,
        0.01,
        ("Positive values indicate lower prediction error than random sampling."),
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1,
        ]
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_final_performance_table(
    results: pd.DataFrame,
    sample_budget: int,
    output_path: str | Path,
    number_of_runs: int = DEFAULT_NUMBER_OF_RUNS,
) -> pd.DataFrame:
    """
    Save a concise table of final strategy performance.

    Confidence-interval columns are also included.
    """

    output_path = Path(output_path)

    final_results = results[results["Samples Used"] == sample_budget].copy()

    if final_results.empty:
        raise ValueError(f"No results found for a sample budget of {sample_budget}.")

    final_results["Log MAE 95% CI"] = calculate_confidence_interval(
        standard_deviation=final_results["Log MAE std"].fillna(0).to_numpy(),
        number_of_runs=number_of_runs,
    )

    final_results["Log RMSE 95% CI"] = calculate_confidence_interval(
        standard_deviation=final_results["Log RMSE std"].fillna(0).to_numpy(),
        number_of_runs=number_of_runs,
    )

    final_results["Log R2 95% CI"] = calculate_confidence_interval(
        standard_deviation=final_results["Log R2 std"].fillna(0).to_numpy(),
        number_of_runs=number_of_runs,
    )

    selected_columns = [
        "Strategy",
        "Samples Used",
        "Log MAE mean",
        "Log MAE std",
        "Log MAE 95% CI",
        "Log RMSE mean",
        "Log RMSE std",
        "Log RMSE 95% CI",
        "Log R2 mean",
        "Log R2 std",
        "Log R2 95% CI",
        "Original MAE mean",
        "Original RMSE mean",
        "Original R2 mean",
    ]

    final_table = (
        final_results[selected_columns]
        .sort_values(
            by="Log RMSE mean",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    final_table.to_csv(
        output_path,
        index=False,
    )

    return final_table


def generate_all_figures(
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    output_directory: str | Path = DEFAULT_FIGURES_DIRECTORY,
    final_sample_budget: int = 200,
    number_of_runs: int = DEFAULT_NUMBER_OF_RUNS,
) -> None:
    """
    Generate all primary and supplementary project figures.
    """

    results = load_active_learning_summary(summary_path)

    (
        primary_directory,
        supplementary_directory,
    ) = prepare_output_directories(output_directory)

    # Primary figure 1: central learning-curve result.
    plot_learning_curve(
        results=results,
        mean_column="Log RMSE mean",
        std_column="Log RMSE std",
        y_axis_label="Log RMSE",
        title="Mean Log RMSE Across Sampling Budgets",
        output_path=(primary_directory / "figure_1_log_rmse_learning_curve.png"),
        number_of_runs=number_of_runs,
    )

    # Primary figure 2: improvement relative to random.
    plot_improvement_over_random(
        results=results,
        output_path=(primary_directory / "figure_2_improvement_over_random.png"),
    )

    # Primary figure 3: endpoint comparison.
    plot_final_strategy_comparison(
        results=results,
        sample_budget=final_sample_budget,
        output_path=(primary_directory / "figure_3_final_strategy_comparison.png"),
        number_of_runs=number_of_runs,
    )

    # Supplementary figure: log MAE.
    plot_learning_curve(
        results=results,
        mean_column="Log MAE mean",
        std_column="Log MAE std",
        y_axis_label="Log MAE",
        title="Mean Log MAE Across Sampling Budgets",
        output_path=(supplementary_directory / "supplementary_log_mae.png"),
        number_of_runs=number_of_runs,
    )

    # Supplementary figure: correctly labeled log R².
    plot_learning_curve(
        results=results,
        mean_column="Log R2 mean",
        std_column="Log R2 std",
        y_axis_label="Log R²",
        title="Log-Scale R² Across Sampling Budgets",
        output_path=(supplementary_directory / "supplementary_log_r2.png"),
        number_of_runs=number_of_runs,
        zero_reference_line=True,
    )

    # Supplementary figure: original-scale RMSE.
    plot_learning_curve(
        results=results,
        mean_column="Original RMSE mean",
        std_column="Original RMSE std",
        y_axis_label="RMSE (pieces/m³)",
        title=("Original-Scale RMSE Across Sampling Budgets"),
        output_path=(supplementary_directory / "supplementary_original_rmse.png"),
        number_of_runs=number_of_runs,
    )

    create_final_performance_table(
        results=results,
        sample_budget=final_sample_budget,
        output_path=(output_directory / "final_performance_table.csv"),
        number_of_runs=number_of_runs,
    )

    print("\nVisualization complete.")

    print(f"Primary figures saved to: {primary_directory}")

    print(f"Supplementary figures saved to: {supplementary_directory}")

    print(
        "Final performance table saved to: "
        f"{output_directory / 'final_performance_table.csv'}"
    )


if __name__ == "__main__":
    generate_all_figures()
