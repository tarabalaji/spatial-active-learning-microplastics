from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Log Microplastics Measurement"
NUMERIC_FEATURES = [
    "Latitude (degree)",
    "Longitude (degree)",
    "Water Sample Depth (m)",
    "Mesh Size (mm)",
    "Year",
    "Month",
]
CATEGORICAL_FEATURES = [
    "Subregion",
    "Sampling Method",
]


def validate_model_data(df: pd.DataFrame) -> None:
    required_columns = [
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
        TARGET_COLUMN,
    ]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Processed dataset is missing required columns: {missing_columns}"
        )


def get_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    validate_model_data(df)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_linear_regression() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )


def build_random_forest(
    random_state: int = 42,
) -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def build_gaussian_process(
    random_state: int = 42,
) -> Pipeline:
    kernel = ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-2, 1e2),
    ) * Matern(
        length_scale=1.0,
        length_scale_bounds=(1e-2, 1e2),
        nu=1.5,
    ) + WhiteKernel(
        noise_level=0.1,
        noise_level_bounds=(1e-5, 1e1),
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def get_baseline_models(
    random_state: int = 42,
) -> dict[str, Pipeline]:
    return {
        "Linear Regression": build_linear_regression(),
        "Random Forest": build_random_forest(random_state=random_state),
        "Gaussian Process": build_gaussian_process(random_state=random_state),
    }


def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    if X_train.empty:
        raise ValueError("X_train cannot be empty.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    model.fit(X_train, y_train)
    return model


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> dict[str, Pipeline]:
    models = get_baseline_models(random_state=random_state)
    trained_models: dict[str, Pipeline] = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")

        trained_models[model_name] = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
        )
        print(f"Finished training {model_name}.")
    return trained_models


def predict_log_concentration(
    model: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    predictions = model.predict(X)
    return np.asarray(predictions)


def convert_log_predictions_to_original_scale(
    log_predictions: np.ndarray,
) -> np.ndarray:
    original_predictions = np.expm1(log_predictions)
    return np.clip(original_predictions, a_min=0, a_max=None)


def predict_original_concentration(
    model: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    log_predictions = predict_log_concentration(model, X)
    return convert_log_predictions_to_original_scale(log_predictions)


def predict_with_uncertainty(
    model: Pipeline,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(
        model.named_steps["model"],
        GaussianProcessRegressor,
    ):
        raise TypeError(
            "Uncertainty estimates are only available for the Gaussian Process model."
        )
    preprocessor = model.named_steps["preprocessor"]
    gaussian_process = model.named_steps["model"]
    X_transformed = preprocessor.transform(X)
    mean_prediction, standard_deviation = gaussian_process.predict(
        X_transformed,
        return_std=True,
    )
    return mean_prediction, standard_deviation


def get_model_parameters(
    model: Pipeline,
) -> dict[str, Any]:
    return model.get_params()
