# End-to-End QSAR Chemical Biodegradation Classification

M.Tech (AIML) — Machine Learning Assignment 2

---

## a. Problem Statement

<!-- TODO (Commit 12): Fill problem statement for QSAR Biodegradation prediction -->

---

## b. Dataset Description

- **Dataset Name:** QSAR Biodegradation Dataset (Mansouri et al., *Journal of Chemical Information and Modeling*, 2013)
- **Source Repository:** [UCI Machine Learning Repository — QSAR Biodegradation (ID: 254)](https://archive.ics.uci.edu/dataset/254/qsar+biodegradation)
- **Number of Instances (Rows):** `1,055` chemical compounds (exceeds minimum requirement of 500 instances)
- **Number of Input Features:** `41` molecular descriptors (`17` continuous topological/matrix eigenvalues and information indices + `24` discrete atom/fragment counts and 2D frequency descriptors; exceeds minimum requirement of 12 features)
- **Missing Values:** `0` (complete dataset with no missing values)
- **Target Variable (`biodeg_class`):** Binary classification indicating experimental environmental biodegradability:
  - `NRB` (**Not Readily Biodegradable** — encoded as `0`): `699` molecules (`66.26%`)
  - `RB` (**Readily Biodegradable** — encoded as `1`): `356` molecules (`33.74%`)
- **Class Balance:** Moderately imbalanced (~2:1 ratio of `NRB` to `RB`), making metrics like **MCC**, **F1**, and **ROC-AUC** particularly informative alongside raw Accuracy.

---

## c. GitHub Repository Link

- **GitHub Repository:** [https://github.com/Infinite-Thinker/BITS_ML_Assignment_2](https://github.com/Infinite-Thinker/BITS_ML_Assignment_2)
- **Live Streamlit App:** <!-- TODO (After Commit 12): Add deployed Streamlit Community Cloud link -->

---

## d. Models Used

### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | TODO | TODO | TODO | TODO | TODO | TODO |
| **Decision Tree** | TODO | TODO | TODO | TODO | TODO | TODO |
| **KNN** | TODO | TODO | TODO | TODO | TODO | TODO |
| **Naive Bayes** | TODO | TODO | TODO | TODO | TODO | TODO |
| **Support Vector Machine** | TODO | TODO | TODO | TODO | TODO | TODO |

### Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | TODO |
| **Decision Tree** | TODO |
| **KNN** | TODO |
| **Naive Bayes** | TODO |
| **Support Vector Machine** | TODO |
| **Overall, Winner for your dataset?** | TODO |
