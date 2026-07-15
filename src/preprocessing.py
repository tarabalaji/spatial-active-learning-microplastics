from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "Microplastics Measurement"
UNUSED_COLUMNS = [
    "ObjectId",
    "x",
    "y",
]
EXPECTED_COLUMNS = [
    "Unique ID",
    "Sample Date",
    "Latitude (degree)",
    "Longitude (degree)",
    "Subregion",
    "Water Sample Depth (m)",
    "Sampling Method",
    "Mesh Size (mm)",
    "Microplastics Measurement",
    "Unit",
    "Concentration Class",
    "Short Reference",
    "DOI",
    "Organization",
]


def load_data(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    df = pd.read_csv(file_path)
    missing_columns = [
        column for column in EXPECTED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop(columns=UNUSED_COLUMNS, errors="ignore")
    cleaned["Sample Date"] = pd.to_datetime(
        cleaned["Sample Date"],
        errors="coerce",
    )
    numeric_columns = [
        "Latitude (degree)",
        "Longitude (degree)",
        "Water Sample Depth (m)",
        "Mesh Size (mm)",
        TARGET_COLUMN,
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )
    required_columns = [
        "Unique ID",
        "Sample Date",
        "Latitude (degree)",
        "Longitude (degree)",
        TARGET_COLUMN,
    ]
    cleaned = cleaned.dropna(subset=required_columns)
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned[cleaned[TARGET_COLUMN] >= 0]
    return cleaned.reset_index(drop=True)


def validate_cleaned_data(df: pd.DataFrame) -> None:
    invalid_latitude = ~df["Latitude (degree)"].between(-90, 90)
    invalid_longitude = ~df["Longitude (degree)"].between(-180, 180)

    if invalid_latitude.any():
        raise ValueError("Invalid latitude values were found.")

    if invalid_longitude.any():
        raise ValueError("Invalid longitude values were found.")

    units = set(df["Unit"].dropna().astype(str).str.strip())

    if units != {"pieces/m3"}:
        raise ValueError(f"Expected only pieces/m3, but found: {sorted(units)}")

    if df["Unique ID"].duplicated().any():
        duplicated_ids = df.loc[
            df["Unique ID"].duplicated(keep=False),
            "Unique ID",
        ].unique()

        raise ValueError(f"Duplicate Unique ID values found: {duplicated_ids[:10]}")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["Year"] = featured["Sample Date"].dt.year
    featured["Month"] = featured["Sample Date"].dt.month
    featured["Season"] = featured["Month"].map(
        {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Autumn",
            10: "Autumn",
            11: "Autumn",
        }
    )
    featured["Log Microplastics Measurement"] = np.log1p(featured[TARGET_COLUMN])
    return featured


def prepare_data(file_path: str | Path) -> pd.DataFrame:
    df = load_data(file_path)
    df = clean_data(df)
    validate_cleaned_data(df)
    df = create_features(df)
    return df


def save_processed_data(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
