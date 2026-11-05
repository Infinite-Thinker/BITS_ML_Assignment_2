"""
EcoMol QSAR — Interactive Chemical Biodegradability Evaluation Portal
Streamlit web application (Commit 8: App skeleton, cached model loader,
sidebar controls, and strict schema validation for uploaded test CSVs).
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
import pandas as pd
import streamlit as st

from preprocess import (
    RAW_DATASET_PATH,
    TARGET_COLUMN,
    LABEL_TO_INT,
    encode_biodeg_target,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_STORE_DIR = PROJECT_ROOT / "model"

MODEL_ARTIFACTS: Dict[str, str] = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


st.set_page_config(
    page_title="EcoMol QSAR | Biodegradability Classifier",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_expected_descriptor_names() -> List[str]:
    """Read the 41 canonical molecular descriptor names from the reference dataset."""
    ref_df = pd.read_csv(RAW_DATASET_PATH, nrows=1)
    return [col for col in ref_df.columns if col != TARGET_COLUMN]


@st.cache_resource
def load_serialized_pipeline(selected_model_name: str):
    """
    Load a pre-trained scikit-learn Pipeline (.joblib) from model/ using
    st.cache_resource so it stays in memory across reruns and NEVER retrains.
    """
    if selected_model_name not in MODEL_ARTIFACTS:
        raise KeyError(f"Unsupported model selection: {selected_model_name}")
    artifact_path = MODEL_STORE_DIR / MODEL_ARTIFACTS[selected_model_name]
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Saved model file not found at '{artifact_path}'. "
            "Run `python train_models.py` first."
        )
    return joblib.load(artifact_path)


def validate_uploaded_test_frame(
    candidate_df: pd.DataFrame,
    expected_descriptors: List[str],
) -> Tuple[bool, str]:
    """
    Validate that the uploaded CSV contains:
      1. At least one row of test compounds
      2. The mandatory target column ('biodeg_class')
      3. All 41 required molecular descriptor input columns
      4. Valid target class values ('NRB'/'RB' or 0/1)
    """
    if candidate_df is None or candidate_df.empty:
        return False, "The uploaded CSV file is empty (0 rows)."

    if TARGET_COLUMN not in candidate_df.columns:
        return (
            False,
            f"Missing required target column `{TARGET_COLUMN}`. "
            "The uploaded test CSV must include both input features and ground-truth labels.",
        )

    missing_descriptors = [
        col for col in expected_descriptors if col not in candidate_df.columns
    ]
    if missing_descriptors:
        preview = ", ".join(missing_descriptors[:8])
        more_txt = (
            f" (and {len(missing_descriptors) - 8} more)"
            if len(missing_descriptors) > 8
            else ""
        )
        return (
            False,
            f"Uploaded file is missing {len(missing_descriptors)} required molecular descriptor "
            f"column(s): `{preview}`{more_txt}.",
        )

    non_numeric_cols = [
        col
        for col in expected_descriptors
        if not pd.api.types.is_numeric_dtype(candidate_df[col])
    ]
    if non_numeric_cols:
        return (
            False,
            f"Non-numeric values detected in descriptor column(s): `{', '.join(non_numeric_cols[:5])}`.",
        )

    try:
        _ = encode_biodeg_target(candidate_df[TARGET_COLUMN])
    except ValueError as err:
        return (
            False,
            f"Invalid values in `{TARGET_COLUMN}`: {err}. Expected `{list(LABEL_TO_INT.keys())}` or `0`/`1`.",
        )

    return True, "Schema validation passed: all 41 descriptors and target labels verified."


def render_sidebar() -> Tuple[str, Optional[pd.DataFrame]]:
    """Render sidebar controls for classifier selection and holdout test CSV upload."""
    with st.sidebar:
        st.markdown("## 🧪 EcoMol QSAR Control Panel")
        st.caption(
            "Select a pre-trained classifier and upload holdout test data (`test_data.csv`) "
            "for zero-retraining inference."
        )
        st.divider()

        chosen_model = st.selectbox(
            "1. Choose Classification Model",
            options=list(MODEL_ARTIFACTS.keys()),
            index=0,
            help="Loads the corresponding pre-trained Pipeline from the model/ directory.",
        )

        uploaded_csv = st.file_uploader(
            "2. Upload Holdout Test CSV (`test_data.csv`)",
            type=["csv"],
            help="Upload test data only (41 molecular descriptors + biodeg_class target column).",
        )

        st.divider()
        st.markdown(
            "**Target Encoding:**\n"
            "- `RB` (`1`): Readily Biodegradable\n"
            "- `NRB` (`0`): Not Readily Biodegradable"
        )

    parsed_df: Optional[pd.DataFrame] = None
    if uploaded_csv is not None:
        try:
            parsed_df = pd.read_csv(uploaded_csv)
        except Exception as read_err:
            st.sidebar.error(f"Could not parse file as CSV: {read_err}")

    return chosen_model, parsed_df


def main() -> None:
    st.title("🧪 EcoMol QSAR: Chemical Biodegradability Screening")
    st.markdown(
        "Evaluate five pre-trained machine learning pipelines on **41 molecular descriptors** "
        "from the UCI QSAR Biodegradation benchmark (*Mansouri et al., 2013*). "
        "Models are loaded directly from `model/` via `st.cache_resource` without retraining."
    )

    expected_descriptors = get_expected_descriptor_names()
    chosen_model_name, uploaded_df = render_sidebar()

    try:
        active_pipeline = load_serialized_pipeline(chosen_model_name)
        st.caption(
            f"**Active Pipeline Loaded:** `{chosen_model_name}` "
            f"(`model/{MODEL_ARTIFACTS[chosen_model_name]}`)"
        )
    except Exception as load_err:
        st.error(f"Failed to load saved model pipeline: {load_err}")
        st.stop()

    if uploaded_df is None:
        st.info(
            "👈 Please upload **`test_data.csv`** using the sidebar file uploader to validate "
            "the schema and run model inference."
        )
        return

    is_valid, validation_msg = validate_uploaded_test_frame(
        uploaded_df, expected_descriptors
    )
    if not is_valid:
        st.error(f"❌ **Dataset Validation Error:** {validation_msg}")
        st.stop()

    st.success(
        f"✅ {validation_msg} Loaded **{len(uploaded_df)}** test compounds for evaluation "
        f"with **{chosen_model_name}**."
    )

    with st.expander("Preview Uploaded Test Compounds (First 10 Rows)", expanded=True):
        st.dataframe(uploaded_df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
