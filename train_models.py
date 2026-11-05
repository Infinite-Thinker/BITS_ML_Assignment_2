"""
Model training and evaluation suite for QSAR Chemical Biodegradation.
Part 1 (Commit 5): Logistic Regression and Decision Tree Classifier,
bundled as full preprocessing + classification Pipelines and saved via joblib.
"""

from pathlib import Path
from typing import Dict, Any
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
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocess import (
    PROJECT_ROOT,
    SPLIT_SEED,
    PartitionedDataset,
    build_descriptor_preprocessor,
)


MODEL_STORE_DIR = PROJECT_ROOT / "model"


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
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
    }


def build_candidate_estimators() -> Dict[str, Dict[str, Any]]:
    """
    Return registry of classification models and their target artifact filenames.
    Part 1 includes Logistic Regression and Decision Tree Classifier.
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
    }


def train_and_persist_models() -> pd.DataFrame:
    """Fit each full Pipeline on X_train, evaluate on X_test, and save to model/."""
    MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = PartitionedDataset(save_holdout_csv=False)

    registry = build_candidate_estimators()
    evaluation_records = []

    print("=" * 78)
    print("QSAR BIODEGRADATION -- MODEL TRAINING & EVALUATION (PART 1)")
    print("=" * 78)

    for model_label, spec in registry.items():
        full_pipe = Pipeline(
            steps=[
                ("descriptor_prep", build_descriptor_preprocessor(X_train)),
                ("classifier", spec["estimator"]),
            ]
        )
        full_pipe.fit(X_train, y_train)

        # Evaluate on training set (to monitor overfitting) and holdout test set
        train_acc = accuracy_score(y_train, full_pipe.predict(X_train))
        y_test_pred = full_pipe.predict(X_test)
        y_test_prob = full_pipe.predict_proba(X_test)
        test_metrics = compute_six_metrics(y_test, y_test_pred, y_test_prob)

        artifact_path = MODEL_STORE_DIR / spec["artifact_name"]
        joblib.dump(full_pipe, artifact_path, compress=3)
        size_kb = artifact_path.stat().st_size / 1024.0

        row = {"ML Model Name": model_label, **test_metrics}
        evaluation_records.append(row)

        print(f"\n[{model_label}]")
        print(f"  Saved Pipeline : {artifact_path.relative_to(PROJECT_ROOT)} ({size_kb:.1f} KB)")
        print(f"  Train Accuracy : {train_acc:.4f} | Holdout Test Accuracy : {test_metrics['Accuracy']:.4f}")
        print(
            f"  Test Metrics   : AUC={test_metrics['AUC']:.4f}, "
            f"Precision={test_metrics['Precision']:.4f}, Recall={test_metrics['Recall']:.4f}, "
            f"F1={test_metrics['F1']:.4f}, MCC={test_metrics['MCC']:.4f}"
        )

    summary_df = pd.DataFrame(evaluation_records)
    print("\n--- Holdout Test Metrics Summary (Part 1) ---")
    print(summary_df.to_string(index=False))
    print("=" * 78)
    return summary_df


if __name__ == "__main__":
    train_and_persist_models()
