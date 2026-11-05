"""
EcoMol QSAR — Interactive Chemical Biodegradability Evaluation Portal
Streamlit web application (Commit 10: All-model comparison bar chart from
results.csv, sample test_data.csv download button, dataset info expander,
multi-tab interface, and custom environmental-chemistry styling).
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
    HOLDOUT_TEST_CSV_PATH,
    TARGET_COLUMN,
    LABEL_TO_INT,
    INT_TO_LABEL,
    encode_biodeg_target,
)
from train_models import RESULTS_CSV_PATH, compute_six_metrics


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_STORE_DIR = PROJECT_ROOT / "model"
ASSETS_DIR = PROJECT_ROOT / "assets"

MODEL_ARTIFACTS: Dict[str, str] = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_COLUMNS: List[str] = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


st.set_page_config(
    page_title="2026AJ05253 | QSAR Biodegradability",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_theme() -> None:
    """Apply custom styling for the EcoMol QSAR workbench."""
    st.markdown(
        """
        <style>
        .ecomol-banner {
            background: linear-gradient(135deg, #0f172a 0%, #134e4a 60%, #0f766e 100%);
            color: #f8fafc;
            padding: 1.35rem 1.6rem;
            border-radius: 12px;
            margin-bottom: 1.1rem;
            border-left: 6px solid #2dd4bf;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.14);
        }
        .ecomol-banner h1 {
            margin: 0 0 0.35rem 0;
            font-size: 1.75rem;
            font-weight: 700;
            color: #ffffff;
        }
        .ecomol-banner p {
            margin: 0;
            font-size: 0.96rem;
            color: #ccfbf1;
            line-height: 1.45;
        }
        div[data-testid="stMetric"] {
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            border-top: 4px solid #0d9488;
            padding: 0.75rem 0.9rem;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }
        div[data-testid="stMetricLabel"] > div {
            font-weight: 600 !important;
            color: #334155 !important;
        }
        div[data-testid="stMetricValue"] > div {
            color: #0f172a !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def get_expected_descriptor_names() -> List[str]:
    """Read the 41 canonical molecular descriptor names from the reference dataset."""
    ref_df = pd.read_csv(RAW_DATASET_PATH, nrows=1)
    return [col for col in ref_df.columns if col != TARGET_COLUMN]


@st.cache_data
def load_sample_test_csv_bytes() -> bytes:
    """Load the holdout test_data.csv bytes for the one-click sidebar download button."""
    if HOLDOUT_TEST_CSV_PATH.exists():
        return HOLDOUT_TEST_CSV_PATH.read_bytes()
    return b""


@st.cache_data
def load_benchmark_results_table() -> pd.DataFrame:
    """Load the pre-computed 5x6 model comparison table from results.csv."""
    if not RESULTS_CSV_PATH.exists():
        raise FileNotFoundError(f"Benchmark table not found at {RESULTS_CSV_PATH}")
    return pd.read_csv(RESULTS_CSV_PATH)


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
        cmap="Teal" if "Teal" in plt.colormaps() else "YlGnBu",
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


def build_all_models_comparison_chart(results_df: pd.DataFrame) -> plt.Figure:
    """Plot a grouped bar chart comparing all 5 classifiers across the 6 metrics."""
    melted = results_df.melt(
        id_vars="ML Model Name",
        value_vars=METRIC_COLUMNS,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=135)
    palette = ["#0f766e", "#2563eb", "#d97706", "#7c3aed", "#dc2626"]
    sns.barplot(
        data=melted,
        x="Metric",
        y="Score",
        hue="ML Model Name",
        palette=palette,
        edgecolor="#1e293b",
        linewidth=0.7,
        ax=ax,
    )

    ax.set_ylim(0.35, 1.05)
    ax.set_title(
        "5-Classifier Performance Comparison Across 6 Evaluation Metrics (Holdout Test Set)",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Evaluation Metric", fontsize=10)
    ax.set_ylabel("Metric Score", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.legend(
        title="Classifier Pipeline",
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        frameon=True,
        fontsize=8.8,
        title_fontsize=9.2,
    )
    sns.despine()
    fig.tight_layout()
    return fig


def render_dataset_overview_expander() -> None:
    """Display an expander summarizing the UCI QSAR Biodegradation dataset and schema."""
    with st.expander("ℹ️ About the QSAR Biodegradation Dataset & 41 Molecular Descriptors", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            "**Dataset Source & Scale**\n"
            "- **Repository:** UCI ML Repository (ID 254)\n"
            "- **Total Instances:** `1,055` chemical compounds\n"
            "- **Stratified Split:** `844` train (`80%`) / `211` test (`20%`)\n"
            "- **Missing Values:** `0`"
        )
        c2.markdown(
            "**Feature Taxonomy (41 Descriptors)**\n"
            "- **17 Continuous Features:** Topological & 2D matrix eigenvalues (`SpMax_L`, `J_Dz_e`, `SM6_L`, `HyWi_B_m`, `C_pct`, etc.)\n"
            "- **24 Discrete Features:** Functional group & atom-pair counts (`nHM`, `nO`, `nN`, `nCb_minus`, `B01_C_Br`, `F04_C_N`, etc.)"
        )
        c3.markdown(
            "**Target Variable (`biodeg_class`)**\n"
            "- **`NRB` (`0`):** Not Readily Biodegradable — `699` (`66.26%`)\n"
            "- **`RB` (`1`):** Readily Biodegradable — `356` (`33.74%`)\n"
            "- **Domain Goal:** Screen synthetic chemicals for environmental persistence before industrial release."
        )


def render_sidebar() -> Tuple[str, Optional[pd.DataFrame]]:
    """Render sidebar controls for model selection, sample CSV download, and test upload."""
    with st.sidebar:
        st.markdown("## 🧪 EcoMol QSAR Controls")
        st.markdown("**BITS Roll No:** `2026AJ05253`")
        st.caption(
            "Select a pre-trained classifier and upload holdout test data (`test_data.csv`) "
            "for zero-retraining evaluation."
        )
        st.divider()

        chosen_model = st.selectbox(
            "1. Select Classification Model",
            options=list(MODEL_ARTIFACTS.keys()),
            index=0,
            help="Loads the corresponding pre-trained scikit-learn Pipeline from model/.",
        )

        uploaded_csv = st.file_uploader(
            "2. Upload Holdout Test CSV (`test_data.csv`)",
            type=["csv"],
            help="Upload test data only (41 molecular descriptors + biodeg_class target column).",
        )

        sample_bytes = load_sample_test_csv_bytes()
        if sample_bytes:
            st.download_button(
                label="⬇️ Download Sample test_data.csv (211 rows)",
                data=sample_bytes,
                file_name="test_data.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download the 20% stratified holdout test set to test the uploader above.",
            )

        st.divider()
        st.markdown(
            "**Target Class Legend:**\n"
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
    inject_custom_theme()

    st.markdown(
        """
        <div class="ecomol-banner">
            <h1>🧪 EcoMol QSAR: Chemical Biodegradability Screening Workbench</h1>
            <p style="margin-bottom: 0.45rem; font-weight: 600; color: #5eead4;">
                BITS Pilani WILP — M.Tech (AIML) Machine Learning Assignment 2 &nbsp;|&nbsp; Roll No: 2026AJ05253
            </p>
            <p>
                End-to-end evaluation of 5 pre-trained classification pipelines across 41 molecular
                descriptors (UCI QSAR Biodegradation benchmark). Models are loaded via
                <code>st.cache_resource</code> and evaluated without retraining.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_dataset_overview_expander()

    expected_descriptors = get_expected_descriptor_names()
    chosen_model_name, uploaded_df = render_sidebar()

    try:
        active_pipeline = load_serialized_pipeline(chosen_model_name)
    except Exception as load_err:
        st.error(f"Failed to load saved model pipeline: {load_err}")
        st.stop()

    tab_eval, tab_compare, tab_eda = st.tabs(
        [
            f"🔬 Active Model Evaluation ({chosen_model_name})",
            "📈 All-Model Benchmark Comparison (5×6)",
            "🧬 EDA & Descriptor Insights",
        ]
    )

    with tab_eval:
        st.caption(
            f"**Loaded Artifact:** `model/{MODEL_ARTIFACTS[chosen_model_name]}` "
            f"| **Pipeline Steps:** `ColumnTransformer(SimpleImputer + StandardScaler)` → `{chosen_model_name}`"
        )

        if uploaded_df is None:
            st.info(
                "👈 **Upload `test_data.csv` in the sidebar** (or click *Download Sample test_data.csv* "
                "in the sidebar first) to compute evaluation metrics, the confusion matrix, and "
                "compound-level biodegradability predictions."
            )
        else:
            is_valid, validation_msg = validate_uploaded_test_frame(
                uploaded_df, expected_descriptors
            )
            if not is_valid:
                st.error(f"❌ **Dataset Validation Error:** {validation_msg}")
                st.stop()

            st.success(
                f"✅ {validation_msg} Evaluated **{len(uploaded_df)}** test compounds "
                f"with **{chosen_model_name}** (zero retraining)."
            )

            metrics_dict, y_true, y_pred, y_prob = evaluate_uploaded_compounds(
                active_pipeline, uploaded_df, expected_descriptors
            )

            st.subheader(f"📊 Evaluation Metrics — {chosen_model_name}")
            metric_cols = st.columns(6)
            for col_ui, m_key in zip(metric_cols, METRIC_COLUMNS):
                col_ui.metric(label=m_key, value=f"{metrics_dict[m_key]:.4f}")

            st.divider()

            col_cm, col_rep = st.columns([1.05, 1.15])
            with col_cm:
                st.subheader("🎯 Confusion Matrix Heatmap")
                cm_fig = build_confusion_matrix_figure(y_true, y_pred, chosen_model_name)
                st.pyplot(cm_fig, use_container_width=False)
                plt.close(cm_fig)

            with col_rep:
                st.subheader("📋 Per-Class Classification Report")
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

            with st.expander("🔍 Inspect Compound-Level Predictions (First 20 Rows)", expanded=False):
                pred_df = uploaded_df.copy()
                pred_df.insert(0, "Predicted_Class", [INT_TO_LABEL[int(p)] for p in y_pred])
                pred_df.insert(1, "RB_Probability", np.round(y_prob[:, 1], 4))
                pred_df.insert(
                    2,
                    "Match",
                    np.where(y_pred == y_true, "✅ Correct", "❌ Misclassified"),
                )
                st.dataframe(pred_df.head(20), use_container_width=True)

    with tab_compare:
        st.subheader("🏆 5-Model Holdout Test Benchmark (`results.csv`)")
        try:
            bench_df = load_benchmark_results_table()
            st.dataframe(
                bench_df.style.format({m: "{:.4f}" for m in METRIC_COLUMNS}).highlight_max(
                    subset=METRIC_COLUMNS, color="#ccfbf1"
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("📊 Visual Comparison Across All 6 Evaluation Metrics")
            comp_fig = build_all_models_comparison_chart(bench_df)
            st.pyplot(comp_fig, use_container_width=True)
            plt.close(comp_fig)
        except Exception as bench_err:
            st.warning(f"Could not load benchmark results table: {bench_err}")

    with tab_eda:
        st.subheader("🧬 Exploratory Data Analysis Visualizations")
        eda_col1, eda_col2 = st.columns(2)
        p_balance = ASSETS_DIR / "class_balance.png"
        p_scale = ASSETS_DIR / "descriptor_scale_disparity.png"
        p_corr = ASSETS_DIR / "top_descriptor_correlations.png"

        with eda_col1:
            if p_balance.exists():
                st.image(str(p_balance), caption="Target Class Distribution (NRB vs RB)", use_container_width=True)
        with eda_col2:
            if p_scale.exists():
                st.image(str(p_scale), caption="Descriptor Dynamic Range Disparity (Log10 Scale)", use_container_width=True)

        if p_corr.exists():
            st.image(
                str(p_corr),
                caption="Inter-Correlation Heatmap of Top 12 Biodegradability-Linked Molecular Descriptors",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
