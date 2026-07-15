from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from src.models import (
    get_features_and_target,
    predict_log_concentration,
    train_all_models,
)


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    X, y = get_features_and_target(df)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def create_spatial_blocks(
    df: pd.DataFrame,
    latitude_block_size: float = 2.0,
    longitude_block_size: float = 2.0,
) -> pd.Series:
    latitude_block = np.floor(df["Latitude (degree)"] / latitude_block_size).astype(int)
    longitude_block = np.floor(df["Longitude (degree)"] / longitude_block_size).astype(
        int
    )
    return latitude_block.astype(str) + "_" + longitude_block.astype(str)


def spatial_split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    latitude_block_size: float = 2.0,
    longitude_block_size: float = 2.0,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    X, y = get_features_and_target(df)
    groups = create_spatial_blocks(
        df=df,
        latitude_block_size=latitude_block_size,
        longitude_block_size=longitude_block_size,
    )
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )
    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()
    y_train = y.iloc[train_indices].copy()
    y_test = y.iloc[test_indices].copy()
    return X_train, X_test, y_train, y_test


def calculate_metrics(
    y_true_log: pd.Series | np.ndarray,
    y_pred_log: np.ndarray,
) -> dict[str, float]:
    y_true_log = np.asarray(y_true_log)
    y_pred_log = np.asarray(y_pred_log)
    log_mae = mean_absolute_error(y_true_log, y_pred_log)
    log_rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    log_r2 = r2_score(y_true_log, y_pred_log)
    y_true_original = np.expm1(y_true_log)
    y_pred_original = np.expm1(y_pred_log)
    y_pred_original = np.clip(
        y_pred_original,
        a_min=0,
        a_max=None,
    )

    original_mae = mean_absolute_error(
        y_true_original,
        y_pred_original,
    )
    original_rmse = np.sqrt(
        mean_squared_error(
            y_true_original,
            y_pred_original,
        )
    )
    original_r2 = r2_score(
        y_true_original,
        y_pred_original,
    )
    return {
        "Log MAE": log_mae,
        "Log RMSE": log_rmse,
        "Log R2": log_r2,
        "Original MAE": original_mae,
        "Original RMSE": original_rmse,
        "Original R2": original_r2,
    }


def evaluate_model(
    model_name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | str]:
    predictions = predict_log_concentration(
        model=model,
        X=X_test,
    )
    metrics = calculate_metrics(
        y_true_log=y_test,
        y_pred_log=predictions,
    )
    return {
        "Model": model_name,
        **metrics,
    }


def evaluate_all_models(
    trained_models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    results = []
    for model_name, model in trained_models.items():
        print(f"Evaluating {model_name}...")
        result = evaluate_model(
            model_name=model_name,
            model=model,
            X_test=X_test,
            y_test=y_test,
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    return results_df.sort_values(
        by="Log RMSE",
        ascending=True,
    ).reset_index(drop=True)


def run_baseline_evaluation(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    use_spatial_split: bool = True,
    latitude_block_size: float = 2.0,
    longitude_block_size: float = 2.0,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:

    if use_spatial_split:
        X_train, X_test, y_train, y_test = spatial_split_data(
            df=df,
            test_size=test_size,
            random_state=random_state,
            latitude_block_size=latitude_block_size,
            longitude_block_size=longitude_block_size,
        )
    else:
        X_train, X_test, y_train, y_test = split_data(
            df=df,
            test_size=test_size,
            random_state=random_state,
        )
    trained_models = train_all_models(
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
    )
    results = evaluate_all_models(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test,
    )
    return trained_models, results


def save_evaluation_results(
    results: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    results.to_csv(
        output_path,
        index=False,
    )


def describe_spatial_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    latitude_block_size: float = 2.0,
    longitude_block_size: float = 2.0,
) -> None:
    X, y = get_features_and_target(df)
    groups = create_spatial_blocks(
        df=df,
        latitude_block_size=latitude_block_size,
        longitude_block_size=longitude_block_size,
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, test_indices = next(splitter.split(X, y, groups=groups))
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    print("\nSpatial split diagnostics")
    print("-------------------------")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")
    print(f"Training blocks: {groups.iloc[train_indices].nunique()}")
    print(f"Testing blocks: {groups.iloc[test_indices].nunique()}")

    print("\nTraining subregions:")
    print(train_df["Subregion"].value_counts())

    print("\nTesting subregions:")
    print(test_df["Subregion"].value_counts())

    print("\nTraining concentration summary:")
    print(train_df["Microplastics Measurement"].describe())

    print("\nTesting concentration summary:")
    print(test_df["Microplastics Measurement"].describe())

    print("\nTraining maximum concentration:")
    print(train_df["Microplastics Measurement"].max())

    print("\nTesting maximum concentration:")
    print(test_df["Microplastics Measurement"].max())


def run_repeated_spatial_evaluation(
    df: pd.DataFrame,
    block_size: float = 1.0,
    seeds: list[int] | None = None,
    test_size: float = 0.2,
) -> pd.DataFrame:

    if seeds is None:
        seeds = [0, 1, 2, 3, 4, 5, 10, 20, 30, 42]
    all_results = []
    for seed in seeds:
        print("\n" + "=" * 60)
        print(f"SPATIAL SPLIT SEED: {seed}")
        print(f"BLOCK SIZE: {block_size}°")
        print("=" * 60)
        _, results = run_baseline_evaluation(
            df=df,
            test_size=test_size,
            random_state=seed,
            use_spatial_split=True,
            latitude_block_size=block_size,
            longitude_block_size=block_size,
        )
        results["Random State"] = seed
        results["Block Size"] = block_size
        all_results.append(results)
    return pd.concat(
        all_results,
        ignore_index=True,
    )


def summarize_repeated_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "Log MAE",
        "Log RMSE",
        "Log R2",
        "Original MAE",
        "Original RMSE",
        "Original R2",
    ]

    summary = (
        results.groupby("Model")[metric_columns].agg(["mean", "std"]).reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else " ".join(part for part in column if part)
        for column in summary.columns
    ]
    return summary
