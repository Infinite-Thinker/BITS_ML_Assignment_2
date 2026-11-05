"""
Data splitting and preprocessing pipeline for QSAR Biodegradation classification.
Performs a leakage-free stratified 80/20 train-test split prior to fitting any
imputation or scaling transformers, and exports the holdout test_data.csv.
"""

from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATASET_PATH = PROJECT_ROOT / "data" / "qsar_biodeg.csv"
HOLDOUT_TEST_CSV_PATH = PROJECT_ROOT / "test_data.csv"

TARGET_COLUMN = "biodeg_class"
POSITIVE_CLASS_LABEL = "RB"   # Readily Biodegradable -> 1
NEGATIVE_CLASS_LABEL = "NRB"  # Not Readily Biodegradable -> 0
LABEL_TO_INT = {NEGATIVE_CLASS_LABEL: 0, POSITIVE_CLASS_LABEL: 1}
INT_TO_LABEL = {0: NEGATIVE_CLASS_LABEL, 1: POSITIVE_CLASS_LABEL}

SPLIT_SEED = 42
TEST_FRACTION = 0.20


def encode_biodeg_target(target_series: pd.Series) -> np.ndarray:
    """Convert string target ('NRB'/'RB') or numeric (0/1) into binary 0/1 array."""
    if pd.api.types.is_numeric_dtype(target_series):
        return target_series.astype(int).to_numpy()
    cleaned = target_series.astype(str).str.strip().str.upper()
    unmapped = set(cleaned.unique()) - set(LABEL_TO_INT.keys())
    if unmapped:
        raise ValueError(f"Unrecognized target class labels found: {unmapped}")
    return cleaned.map(LABEL_TO_INT).astype(int).to_numpy()


def PartitionedDataset(
    csv_path: Path = RAW_DATASET_PATH,
    save_holdout_csv: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load raw QSAR dataset, execute a stratified 80/20 split BEFORE any
    feature transformation, and optionally write holdout test_data.csv.
    """
    mol_df = pd.read_csv(csv_path)
    if TARGET_COLUMN not in mol_df.columns:
        raise KeyError(f"Expected target column '{TARGET_COLUMN}' not found in {csv_path}")

    feature_cols = [col for col in mol_df.columns if col != TARGET_COLUMN]
    X_all = mol_df[feature_cols].copy()
    y_raw = mol_df[TARGET_COLUMN].copy()
    y_all = encode_biodeg_target(y_raw)

    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X_all,
        y_all,
        y_raw,
        test_size=TEST_FRACTION,
        stratify=y_all,
        random_state=SPLIT_SEED,
    )

    if save_holdout_csv:
        holdout_df = X_test.copy()
        holdout_df[TARGET_COLUMN] = y_raw_test.values
        holdout_df.to_csv(HOLDOUT_TEST_CSV_PATH, index=False)

    return X_train, X_test, y_train, y_test


def build_descriptor_preprocessor(reference_features: pd.DataFrame) -> ColumnTransformer:
    """
    Construct a ColumnTransformer separating continuous topological indices,
    discrete atom/fragment count descriptors, and any categorical columns.
    Must be fit ONLY on X_train inside each model's Pipeline.
    """
    continuous_cols: List[str] = reference_features.select_dtypes(
        include=["float64", "float32"]
    ).columns.tolist()
    discrete_count_cols: List[str] = reference_features.select_dtypes(
        include=["int64", "int32"]
    ).columns.tolist()
    categorical_cols: List[str] = reference_features.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    continuous_branch = Pipeline(
        steps=[
            ("median_imputer", SimpleImputer(strategy="median")),
            ("zscore_scaler", StandardScaler()),
        ]
    )

    discrete_branch = Pipeline(
        steps=[
            ("mode_imputer", SimpleImputer(strategy="most_frequent")),
            ("zscore_scaler", StandardScaler()),
        ]
    )

    transformers = [
        ("cont_descriptors", continuous_branch, continuous_cols),
        ("disc_descriptors", discrete_branch, discrete_count_cols),
    ]

    if categorical_cols:
        categorical_branch = Pipeline(
            steps=[
                ("const_imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat_features", categorical_branch, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,  # Always output dense matrix (required by GaussianNB)
    )


def main() -> None:
    X_train, X_test, y_train, y_test = PartitionedDataset(save_holdout_csv=True)

    # Fit preprocessor strictly on X_train to verify zero-leakage transformation
    preprocessor = build_descriptor_preprocessor(X_train)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    train_rb_pct = (y_train.sum() / len(y_train)) * 100.0
    test_rb_pct = (y_test.sum() / len(y_test)) * 100.0
    test_csv_kb = HOLDOUT_TEST_CSV_PATH.stat().st_size / 1024.0

    print("=" * 72)
    print("QSAR BIODEGRADATION -- SPLIT & PREPROCESSING VERIFICATION")
    print("=" * 72)
    print(f"Stratified Split (80/20, random_state={SPLIT_SEED}):")
    print(f"  Training Set   : {X_train.shape[0]:4d} molecules x {X_train.shape[1]} features "
          f"(RB={y_train.sum()} [{train_rb_pct:.2f}%], NRB={len(y_train) - y_train.sum()} [{100 - train_rb_pct:.2f}%])")
    print(f"  Holdout Test   : {X_test.shape[0]:4d} molecules x {X_test.shape[1]} features "
          f"(RB={y_test.sum()} [{test_rb_pct:.2f}%], NRB={len(y_test) - y_test.sum()} [{100 - test_rb_pct:.2f}%])")
    print("\nColumnTransformer Output Verification (fit on X_train only):")
    print(f"  X_train_scaled : shape={X_train_scaled.shape}, mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
    print(f"  X_test_scaled  : shape={X_test_scaled.shape}, mean={X_test_scaled.mean():.4f}, std={X_test_scaled.std():.4f}")
    print(f"\nExported Holdout Test CSV:")
    print(f"  Path           : {HOLDOUT_TEST_CSV_PATH.relative_to(PROJECT_ROOT)} ({test_csv_kb:.1f} KB, {len(X_test)} rows)")
    print("=" * 72)


if __name__ == "__main__":
    main()
