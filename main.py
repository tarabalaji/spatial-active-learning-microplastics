from pathlib import Path

from src.active_learning import (
    run_repeated_active_learning,
    summarize_active_learning_results,
)
from src.evaluation import (
    describe_spatial_split,
    run_repeated_spatial_evaluation,
    save_evaluation_results,
    spatial_split_data,
    summarize_repeated_results,
)
from src.preprocessing import prepare_data, save_processed_data
from src.statistics import run_statistical_analysis
from src.visualization import generate_all_figures

DATA_PATH = Path("data/raw/mediterranean_microplastics.csv")
PROCESSED_PATH = Path("data/processed/mediterranean_microplastics_processed.csv")


def main() -> None:
    df = prepare_data(DATA_PATH)
    print("\nDataset successfully loaded and prepared.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nDate range:")
    print(f"{df['Sample Date'].min().date()} to {df['Sample Date'].max().date()}")
    print("\nSampling methods:")
    print(df["Sampling Method"].value_counts())
    print("\nConcentration summary:")
    print(df["Microplastics Measurement"].describe())
    print("\nUnits:")
    print(df["Unit"].value_counts())
    print("\nMissing values:")
    print(df.isna().sum()[df.isna().sum() > 0])

    save_processed_data(
        df=df,
        output_path=PROCESSED_PATH,
    )
    print(f"\nProcessed data saved to: {PROCESSED_PATH}")
    describe_spatial_split(
        df=df,
        test_size=0.2,
        random_state=42,
        latitude_block_size=1.0,
        longitude_block_size=1.0,
    )

    print("\nStarting repeated spatial evaluation...")

    repeated_results = run_repeated_spatial_evaluation(
        df=df,
        block_size=1.0,
        seeds=[0, 1, 2, 3, 4, 5, 10, 20, 30, 42],
        test_size=0.2,
    )

    summary_results = summarize_repeated_results(repeated_results)
    print("\nPreparing repeated active learning experiment...")
    X_pool, X_test, y_pool, y_test = spatial_split_data(
        df=df,
        test_size=0.2,
        random_state=42,
        latitude_block_size=1.0,
        longitude_block_size=1.0,
    )

    repeated_active_results = run_repeated_active_learning(
        X_pool=X_pool,
        y_pool=y_pool,
        X_test=X_test,
        y_test=y_test,
        seeds=[0, 1, 2, 3, 4, 5, 10, 20, 30, 42],
        initial_sample_size=30,
        maximum_sample_size=200,
        batch_size=10,
        uncertainty_weight=0.3,
        geographic_weight=0.7,
    )
    active_summary = summarize_active_learning_results(repeated_active_results)
    print("\nRepeated active learning summary:")
    print(active_summary.to_string(index=False))

    save_evaluation_results(
        results=repeated_active_results,
        output_path=Path("results/repeated_active_learning_metrics.csv"),
    )
    save_evaluation_results(
        results=active_summary,
        output_path=Path("results/repeated_active_learning_summary.csv"),
    )
    print("\nRepeated active learning results saved in the results folder.")
    print("\nRepeated spatial evaluation summary:")
    print(summary_results.to_string(index=False))

    save_evaluation_results(
        results=repeated_results,
        output_path=Path("results/repeated_spatial_metrics.csv"),
    )
    save_evaluation_results(
        results=summary_results,
        output_path=Path("results/repeated_spatial_summary.csv"),
    )
    print("\nRepeated spatial results saved in the results folder.")
    print("\nGenerating figures...")

    generate_all_figures(
        summary_path=Path("results/repeated_active_learning_summary.csv"),
        output_directory=Path("figures"),
        final_sample_budget=200,
    )

    print("\nRunning statistical analysis...")

    run_statistical_analysis(
        results_path=Path("results/repeated_active_learning_metrics.csv"),
        output_directory=Path("results/statistics"),
        final_sample_budget=200,
        alpha=0.05,
    )


if __name__ == "__main__":
    main()
