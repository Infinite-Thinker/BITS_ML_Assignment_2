"""
EcoMol QSAR — Interactive Chemical Biodegradability Evaluation Portal
Streamlit web application (Commit 9: Zero-retraining inference on uploaded test
compounds, 6-metric st.metric cards, confusion matrix heatmap, and classification report).
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import (
    RAW_DATASET_PATH,
    TARGET_COLUMN,
    LABEL_TO_INT,
    INT_TO_LABEL,
    encode_biodeg_target,
)
from train_models import compute_six_metrics


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


def evaluate_uploaded_compounds(
    pipeline,
    uploaded_df: pd.DataFrame,
    expected_descriptors: List[str],
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference ONLY (predict and predict_proba) using the loaded pipeline
    and compute the 6 mandatory evaluation metrics.
    """
    X_eval = uploaded_df[expected_descriptors].copy()
    y_true = encode_biodeg_target(uploaded_df[TARGET_COLUMN])

    y_pred = pipeline.predict(X_eval)
    y_prob = pipeline.predict_proba(X_eval)
    metrics_dict = compute_six_metrics(y_true, y_pred, y_prob)

    return metrics_dict, y_true, y_pred, y_prob


def build_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_title: str,
) -> plt.Figure:
    """Create an annotated confusion matrix heatmap for NRB (0) vs RB (1)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    total = cm.sum()
    annot_labels = np.empty_like(cm, dtype=object)
    quadrant_names = [["True NRB (TN)", "False RB (FP)"], ["False NRB (FN)", "True RB (TP)"]]

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = (count / total) * 100.0 if total > 0 else 0.0
            annot_labels[i, j] = f"{quadrant_names[i][j]}\n{count} ({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=130)
    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred: NRB (0)", "Pred: RB (1)"],
        yticklabels=["Actual: NRB (0)", "Actual: RB (1)"],
        linewidths=1.5,
        linecolor="white",
        annot_kws={"size": 10, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_title}", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Model Predicted Biodegradability", fontsize=9.5)
    ax.set_ylabel("Experimental Ground Truth", fontsize=9.5)
    fig.tight_layout()
    return fig


def build_classification_report_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Return a clean DataFrame representation of sklearn's classification_report."""
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["NRB (Not Readily Biodeg.)", "RB (Readily Biodeg.)"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    if "accuracy" in report_df.index:
        report_df = report_df.drop(index="accuracy")
    report_df["support"] = report_df["support"].astype(int)
    return report_df.round(4)


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
            "👈 Please upload **`test_data.csv`** using the sidebar file uploader to compute "
            "the 6 evaluation metrics, confusion matrix, and classification report."
        )
        return

    is_valid, validation_msg = validate_uploaded_test_frame(
        uploaded_df, expected_descriptors
    )
    if not is_valid:
        st.error(f"❌ **Dataset Validation Error:** {validation_msg}")
        st.stop()

    st.success(
        f"✅ {validation_msg} Evaluated **{len(uploaded_df)}** test compounds "
        f"using **{chosen_model_name}** (zero retraining)."
    )

    metrics_dict, y_true, y_pred, y_prob = evaluate_uploaded_compounds(
        active_pipeline, uploaded_df, expected_descriptors
    )

    # Section 1: 6 Mandatory Evaluation Metrics (st.metric cards)
    st.subheader(f"📊 Evaluation Metrics — {chosen_model_name}")
    metric_cols = st.columns(6)
    metric_order = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    for col_ui, m_key in zip(metric_cols, metric_order):
        col_ui.metric(label=m_key, value=f"{metrics_dict[m_key]:.4f}")

    st.divider()

    # Section 2: Confusion Matrix Heatmap & Per-Class Classification Report
    col_cm, col_rep = st.columns([1.05, 1.15])
    with col_cm:
        st.subheader("🎯 Confusion Matrix")
        cm_fig = build_confusion_matrix_figure(y_true, y_pred, chosen_model_name)
        st.pyplot(cm_fig, use_container_width=False)
        plt.close(cm_fig)

    with col_rep:
        st.subheader("📋 Detailed Classification Report")
        report_df = build_classification_report_table(y_true, y_pred)
        st.dataframe(
            report_df.style.format(
                {
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1-score": "{:.4f}",
                    "support": "{:d}",
                }
            ),
            use_container_width=True,
        )

    with st.expander("Preview Uploaded Test Compounds & Predictions (First 15 Rows)", expanded=False):
        preview_df = uploaded_df.copy()
        preview_df.insert(0, "Predicted_Class", [INT_TO_LABEL[int(p)] for p in y_pred])
        preview_df.insert(1, "RB_Probability", np.round(y_prob[:, 1], 4))
        st.dataframe(preview_df.head(15), use_container_width=True)


if __name__ == "__main__":
    main()
