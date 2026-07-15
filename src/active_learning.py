from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluation import calculate_metrics
from src.models import (
    build_gaussian_process,
    predict_log_concentration,
    predict_with_uncertainty,
)

SamplingStrategy = Literal[
    "random",
    "uncertainty",
    "geographic",
    "hybrid",
]


def choose_initial_samples(
    pool_size: int,
    initial_sample_size: int,
    random_state: int,
) -> np.ndarray:
    if pool_size <= 0:
        raise ValueError("pool_size must be greater than zero.")

    if initial_sample_size <= 0:
        raise ValueError("initial_sample_size must be greater than zero.")

    if initial_sample_size >= pool_size:
        raise ValueError("initial_sample_size must be smaller than the candidate pool.")

    rng = np.random.default_rng(random_state)

    return rng.choice(
        pool_size,
        size=initial_sample_size,
        replace=False,
    )


def normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return values

    minimum = np.min(values)
    maximum = np.max(values)

    if np.isclose(minimum, maximum):
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


def haversine_distance_matrix(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> np.ndarray:

    points_a = np.asarray(points_a, dtype=float)
    points_b = np.asarray(points_b, dtype=float)

    if points_a.ndim != 2 or points_a.shape[1] != 2:
        raise ValueError("points_a must have shape (number_of_points, 2).")

    if points_b.ndim != 2 or points_b.shape[1] != 2:
        raise ValueError("points_b must have shape (number_of_points, 2).")

    earth_radius_km = 6371.0

    latitude_a = np.radians(points_a[:, 0])[:, None]
    longitude_a = np.radians(points_a[:, 1])[:, None]

    latitude_b = np.radians(points_b[:, 0])[None, :]
    longitude_b = np.radians(points_b[:, 1])[None, :]

    latitude_difference = latitude_b - latitude_a
    longitude_difference = longitude_b - longitude_a

    haversine_value = (
        np.sin(latitude_difference / 2.0) ** 2
        + np.cos(latitude_a)
        * np.cos(latitude_b)
        * np.sin(longitude_difference / 2.0) ** 2
    )

    haversine_value = np.clip(
        haversine_value,
        0.0,
        1.0,
    )

    angular_distance = 2.0 * np.arcsin(np.sqrt(haversine_value))

    return earth_radius_km * angular_distance


def select_random_samples(
    unlabeled_indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(unlabeled_indices) == 0:
        return np.array([], dtype=int)

    actual_batch_size = min(
        batch_size,
        len(unlabeled_indices),
    )

    return rng.choice(
        unlabeled_indices,
        size=actual_batch_size,
        replace=False,
    )


def select_uncertain_samples(
    model: Pipeline,
    X_pool: pd.DataFrame,
    unlabeled_indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if len(unlabeled_indices) == 0:
        return np.array([], dtype=int)

    actual_batch_size = min(
        batch_size,
        len(unlabeled_indices),
    )

    X_unlabeled = X_pool.iloc[unlabeled_indices]

    _, uncertainty = predict_with_uncertainty(
        model=model,
        X=X_unlabeled,
    )

    ranked_positions = np.argsort(uncertainty)[::-1]

    selected_positions = ranked_positions[:actual_batch_size]

    return unlabeled_indices[selected_positions]


def select_geographic_samples(
    X_pool: pd.DataFrame,
    labeled_indices: np.ndarray,
    unlabeled_indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if len(unlabeled_indices) == 0:
        return np.array([], dtype=int)

    actual_batch_size = min(
        batch_size,
        len(unlabeled_indices),
    )

    coordinate_columns = [
        "Latitude (degree)",
        "Longitude (degree)",
    ]

    coordinates = X_pool[coordinate_columns].to_numpy()

    selected_indices: list[int] = []

    remaining_indices = unlabeled_indices.copy()
    current_labeled_indices = labeled_indices.copy()

    for _ in range(actual_batch_size):
        remaining_coordinates = coordinates[remaining_indices]

        labeled_coordinates = coordinates[current_labeled_indices]

        distance_matrix = haversine_distance_matrix(
            points_a=remaining_coordinates,
            points_b=labeled_coordinates,
        )

        nearest_labeled_distance = distance_matrix.min(axis=1)

        best_position = int(np.argmax(nearest_labeled_distance))

        chosen_index = int(remaining_indices[best_position])

        selected_indices.append(chosen_index)

        current_labeled_indices = np.append(
            current_labeled_indices,
            chosen_index,
        )

        remaining_indices = np.delete(
            remaining_indices,
            best_position,
        )

    return np.asarray(
        selected_indices,
        dtype=int,
    )


def select_hybrid_samples(
    model: Pipeline,
    X_pool: pd.DataFrame,
    labeled_indices: np.ndarray,
    unlabeled_indices: np.ndarray,
    batch_size: int,
    uncertainty_weight: float = 0.3,
    geographic_weight: float = 0.7,
) -> np.ndarray:

    if len(unlabeled_indices) == 0:
        return np.array([], dtype=int)

    if uncertainty_weight < 0 or geographic_weight < 0:
        raise ValueError("Hybrid weights cannot be negative.")

    if not np.isclose(
        uncertainty_weight + geographic_weight,
        1.0,
    ):
        raise ValueError("uncertainty_weight and geographic_weight must add up to 1.")

    actual_batch_size = min(
        batch_size,
        len(unlabeled_indices),
    )

    coordinate_columns = [
        "Latitude (degree)",
        "Longitude (degree)",
    ]

    coordinates = X_pool[coordinate_columns].to_numpy()

    selected_indices: list[int] = []

    remaining_indices = unlabeled_indices.copy()
    current_labeled_indices = labeled_indices.copy()

    for _ in range(actual_batch_size):
        X_remaining = X_pool.iloc[remaining_indices]

        _, uncertainty = predict_with_uncertainty(
            model=model,
            X=X_remaining,
        )

        remaining_coordinates = coordinates[remaining_indices]

        labeled_coordinates = coordinates[current_labeled_indices]

        distance_matrix = haversine_distance_matrix(
            points_a=remaining_coordinates,
            points_b=labeled_coordinates,
        )

        nearest_labeled_distance = distance_matrix.min(axis=1)

        normalized_uncertainty = normalize_scores(uncertainty)

        normalized_distance = normalize_scores(nearest_labeled_distance)

        hybrid_score = (
            uncertainty_weight * normalized_uncertainty
            + geographic_weight * normalized_distance
        )

        best_position = int(np.argmax(hybrid_score))

        chosen_index = int(remaining_indices[best_position])

        selected_indices.append(chosen_index)

        current_labeled_indices = np.append(
            current_labeled_indices,
            chosen_index,
        )

        remaining_indices = np.delete(
            remaining_indices,
            best_position,
        )

    return np.asarray(
        selected_indices,
        dtype=int,
    )


def select_next_batch(
    strategy: SamplingStrategy,
    model: Pipeline,
    X_pool: pd.DataFrame,
    labeled_indices: np.ndarray,
    unlabeled_indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    uncertainty_weight: float = 0.3,
    geographic_weight: float = 0.7,
) -> np.ndarray:

    if strategy == "random":
        return select_random_samples(
            unlabeled_indices=unlabeled_indices,
            batch_size=batch_size,
            rng=rng,
        )

    if strategy == "uncertainty":
        return select_uncertain_samples(
            model=model,
            X_pool=X_pool,
            unlabeled_indices=unlabeled_indices,
            batch_size=batch_size,
        )

    if strategy == "geographic":
        return select_geographic_samples(
            X_pool=X_pool,
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            batch_size=batch_size,
        )

    if strategy == "hybrid":
        return select_hybrid_samples(
            model=model,
            X_pool=X_pool,
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            batch_size=batch_size,
            uncertainty_weight=uncertainty_weight,
            geographic_weight=geographic_weight,
        )

    raise ValueError(f"Unknown sampling strategy: {strategy}")


def evaluate_active_learning_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:

    predictions = predict_log_concentration(
        model=model,
        X=X_test,
    )

    return calculate_metrics(
        y_true_log=y_test,
        y_pred_log=predictions,
    )


def run_active_learning(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategy: SamplingStrategy,
    initial_sample_size: int = 30,
    maximum_sample_size: int = 200,
    batch_size: int = 10,
    random_state: int = 42,
    verbose: bool = False,
    uncertainty_weight: float = 0.3,
    geographic_weight: float = 0.7,
) -> pd.DataFrame:

    if len(X_pool) != len(y_pool):
        raise ValueError("X_pool and y_pool must contain the same number of rows.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must contain the same number of rows.")

    if initial_sample_size <= 0:
        raise ValueError("initial_sample_size must be greater than zero.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if maximum_sample_size > len(X_pool):
        maximum_sample_size = len(X_pool)

    if maximum_sample_size <= initial_sample_size:
        raise ValueError(
            "maximum_sample_size must be greater than initial_sample_size."
        )

    X_pool = X_pool.reset_index(drop=True)
    y_pool = y_pool.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    rng = np.random.default_rng(random_state)

    labeled_indices = choose_initial_samples(
        pool_size=len(X_pool),
        initial_sample_size=initial_sample_size,
        random_state=random_state,
    )

    all_indices = np.arange(len(X_pool))

    results: list[dict[str, float | int | str]] = []

    while len(labeled_indices) <= maximum_sample_size:
        X_labeled = X_pool.iloc[labeled_indices]

        y_labeled = y_pool.iloc[labeled_indices]

        model = build_gaussian_process(random_state=random_state)

        model.fit(
            X_labeled,
            y_labeled,
        )

        metrics = evaluate_active_learning_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(
            {
                "Strategy": strategy,
                "Random State": random_state,
                "Samples Used": len(labeled_indices),
                "Uncertainty Weight": (
                    uncertainty_weight if strategy == "hybrid" else np.nan
                ),
                "Geographic Weight": (
                    geographic_weight if strategy == "hybrid" else np.nan
                ),
                **metrics,
            }
        )

        if verbose:
            print(
                f"{strategy.title()} sampling | "
                f"Samples: {len(labeled_indices)} | "
                f"Log RMSE: {metrics['Log RMSE']:.4f} | "
                f"Log R2: {metrics['Log R2']:.4f}"
            )

        if len(labeled_indices) >= maximum_sample_size:
            break

        unlabeled_indices = np.setdiff1d(
            all_indices,
            labeled_indices,
            assume_unique=False,
        )

        next_indices = select_next_batch(
            strategy=strategy,
            model=model,
            X_pool=X_pool,
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            batch_size=batch_size,
            rng=rng,
            uncertainty_weight=uncertainty_weight,
            geographic_weight=geographic_weight,
        )

        if len(next_indices) == 0:
            break

        labeled_indices = np.concatenate(
            [
                labeled_indices,
                next_indices,
            ]
        )

    return pd.DataFrame(results)


def compare_active_learning_strategies(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    initial_sample_size: int = 30,
    maximum_sample_size: int = 200,
    batch_size: int = 10,
    random_state: int = 42,
    verbose: bool = False,
    uncertainty_weight: float = 0.3,
    geographic_weight: float = 0.7,
) -> pd.DataFrame:
    strategies: list[SamplingStrategy] = [
        "random",
        "uncertainty",
        "geographic",
        "hybrid",
    ]

    all_results = []

    for strategy in strategies:
        print(f"Running {strategy} sampling for seed {random_state}...")

        strategy_results = run_active_learning(
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            strategy=strategy,
            initial_sample_size=initial_sample_size,
            maximum_sample_size=maximum_sample_size,
            batch_size=batch_size,
            random_state=random_state,
            verbose=verbose,
            uncertainty_weight=uncertainty_weight,
            geographic_weight=geographic_weight,
        )

        all_results.append(strategy_results)

    return pd.concat(
        all_results,
        ignore_index=True,
    )


def run_repeated_active_learning(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seeds: list[int] | None = None,
    initial_sample_size: int = 30,
    maximum_sample_size: int = 200,
    batch_size: int = 10,
    verbose: bool = False,
    uncertainty_weight: float = 0.4,
    geographic_weight: float = 0.6,
) -> pd.DataFrame:
    if seeds is None:
        seeds = [
            0,
            1,
            2,
            3,
            4,
            5,
            10,
            20,
            30,
            42,
        ]

    if len(seeds) == 0:
        raise ValueError("At least one random seed must be supplied.")

    all_results = []

    for seed_number, seed in enumerate(
        seeds,
        start=1,
    ):
        print(f"\nActive-learning run {seed_number}/{len(seeds)} (seed {seed})")

        seed_results = compare_active_learning_strategies(
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            initial_sample_size=initial_sample_size,
            maximum_sample_size=maximum_sample_size,
            batch_size=batch_size,
            random_state=seed,
            verbose=verbose,
            uncertainty_weight=uncertainty_weight,
            geographic_weight=geographic_weight,
        )

        all_results.append(seed_results)

    return pd.concat(
        all_results,
        ignore_index=True,
    )


def summarize_active_learning_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate mean and standard deviation at each sampling budget."""

    required_columns = [
        "Strategy",
        "Samples Used",
        "Log MAE",
        "Log RMSE",
        "Log R2",
        "Original MAE",
        "Original RMSE",
        "Original R2",
    ]

    missing_columns = [
        column for column in required_columns if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Active-learning results are missing required columns: {missing_columns}"
        )

    metric_columns = [
        "Log MAE",
        "Log RMSE",
        "Log R2",
        "Original MAE",
        "Original RMSE",
        "Original R2",
    ]

    summary = (
        results.groupby(
            [
                "Strategy",
                "Samples Used",
            ]
        )[metric_columns]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    summary.columns = [
        column if isinstance(column, str) else " ".join(part for part in column if part)
        for column in summary.columns
    ]

    return summary.sort_values(
        by=[
            "Samples Used",
            "Strategy",
        ]
    ).reset_index(drop=True)
