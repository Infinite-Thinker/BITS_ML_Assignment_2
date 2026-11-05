"""
Model training, persistence, and comparative evaluation suite for
QSAR Chemical Biodegradation (all 5 required classification models).
Saves compressed Pipeline artifacts into model/ and exports results.csv.
"""

from pathlib import Path
from typing import Dict, Any
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocess import (
    PROJECT_ROOT,
    SPLIT_SEED,
    PartitionedDataset,
    build_descriptor_preprocessor,
)

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_STORE_DIR = PROJECT_ROOT / "model"
RESULTS_CSV_PATH = PROJECT_ROOT / "results.csv"


def compute_six_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate the 6 mandatory evaluation metrics:
    Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC).
    Supports binary classification (positive class probability y_prob[:, 1])
    as well as multi-class fallback (weighted averages + OVR ROC-AUC).
    """
    unique_classes = np.unique(y_true)
    is_binary = len(unique_classes) == 2

    acc = float(accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    if is_binary:
        pos_scores = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        auc = float(roc_auc_score(y_true, pos_scores))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
    else:
        auc = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        )
        prec = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0) if is_binary else f1, 4),
        "MCC": round(mcc, 4),
    }


def build_candidate_estimators() -> Dict[str, Dict[str, Any]]:
    """
    Return registry of all 5 required classification models and their artifact names:
      1. Logistic Regression
      2. Decision Tree
      3. KNN
      4. Naive Bayes (GaussianNB)
      5. Support Vector Machine (SVC with probability=True for ROC-AUC)
    """
    return {
        "Logistic Regression": {
            "artifact_name": "logistic_regression.joblib",
            "estimator": LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=SPLIT_SEED,
            ),
        },
        "Decision Tree": {
            "artifact_name": "decision_tree.joblib",
            "estimator": DecisionTreeClassifier(
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                random_state=SPLIT_SEED,
            ),
        },
        "KNN": {
            "artifact_name": "knn.joblib",
            "estimator": KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                metric="minkowski",
                p=2,
            ),
        },
        "Naive Bayes": {
            "artifact_name": "naive_bayes.joblib",
            "estimator": GaussianNB(
                var_smoothing=1e-9,
            ),
        },
        "Support Vector Machine": {
            "artifact_name": "svm.joblib",
            "estimator": SVC(
                C=1.5,
                kernel="rbf",
                gamma="scale",
                probability=True,
                random_state=SPLIT_SEED,
            ),
        },
    }


def train_and_persist_models() -> pd.DataFrame:
    """Fit all 5 full Pipelines on X_train, evaluate on X_test, save models and results.csv."""
    MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = PartitionedDataset(save_holdout_csv=False)

    registry = build_candidate_estimators()
    evaluation_records = []

    print("=" * 80)
    print("QSAR BIODEGRADATION -- 5-MODEL BENCHMARK & COMPARISON TABLE")
    print("=" * 80)

    for model_label, spec in registry.items():
        full_pipe = Pipeline(
            steps=[
                ("descriptor_prep", build_descriptor_preprocessor(X_train)),
                ("classifier", spec["estimator"]),
            ]
        )
        full_pipe.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, full_pipe.predict(X_train))
        y_test_pred = full_pipe.predict(X_test)
        y_test_prob = full_pipe.predict_proba(X_test)
        test_metrics = compute_six_metrics(y_test, y_test_pred, y_test_prob)

        artifact_path = MODEL_STORE_DIR / spec["artifact_name"]
        joblib.dump(full_pipe, artifact_path, compress=3)

        row = {"ML Model Name": model_label, **test_metrics}
        evaluation_records.append(row)

        print(
            f"{model_label:24s} | Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_metrics['Accuracy']:.4f} | AUC: {test_metrics['AUC']:.4f} | "
            f"MCC: {test_metrics['MCC']:.4f}"
        )

    summary_df = pd.DataFrame(evaluation_records)
    summary_df.to_csv(RESULTS_CSV_PATH, index=False)

    print("\n--- Final 5x6 Evaluation Metrics Table (Saved to results.csv) ---")
    print(summary_df.to_string(index=False))
    print("=" * 80)
    return summary_df


if __name__ == "__main__":
    train_and_persist_models()
