# QSAR Chemical Biodegradation Classification

**Course:** M.Tech (AIML) — Work Integrated Learning Programmes (WILP)  
**Subject:** Machine Learning — Assignment 2  

---

## a. Problem Statement

Many synthetic chemicals used in industry, agriculture, and pharma do not break down easily in the environment and end up polluting water and soil. Testing every new chemical compound in a wet lab to check if it is biodegradable takes around 28 days and costs a lot of time and money.

In this assignment, the goal is to build a machine learning classification pipeline that predicts whether a chemical molecule is **Readily Biodegradable (`RB`)** or **Not Readily Biodegradable (`NRB`)** using **41 molecular descriptors** calculated from its chemical structure. I have trained and compared 5 required classification models (**Logistic Regression**, **Decision Tree**, **KNN**, **Gaussian Naive Bayes**, and **Support Vector Machine**) on the same train-test split, evaluated them using 6 metrics (**Accuracy**, **AUC**, **Precision**, **Recall**, **F1**, and **MCC**), and built a **Streamlit** web application that loads the saved models to evaluate uploaded test CSV files without retraining.

---

## b. Dataset Description

- **Dataset Name:** QSAR Biodegradation Dataset (Mansouri et al., 2013)
- **Source Link:** [UCI Machine Learning Repository — QSAR Biodegradation (ID: 254)](https://archive.ics.uci.edu/dataset/254/qsar+biodegradation)
- **Total Rows (Instances):** `1,055` chemical compounds (meets the requirement of $\ge 500$ instances)
- **Total Input Features:** `41` molecular descriptors (meets the requirement of $\ge 12$ features):
  - **17 Continuous Features (`float64`):** Topological and matrix-based molecular indices such as `SpMax_L`, `J_Dz_e`, `C_pct` (carbon percentage), `SM6_L`, `HyWi_B_m`, `LOC`, `Me`, `Mi`, etc.
  - **24 Discrete Features (`int64`):** Atom and functional group counts such as `nHM` (heavy atoms), `nO` (oxygen atoms), `nN` (nitrogen atoms), `nCb_minus` (substituted benzene rings), `B01_C_Br`, `F04_C_N`, etc.
- **Missing Values:** `0` (there are no missing or null values in the dataset)
- **Target Column (`biodeg_class`):** Binary classification label:
  - **`NRB` (encoded as `0`):** Not Readily Biodegradable — `699` rows (`66.26%`)
  - **`RB` (encoded as `1`):** Readily Biodegradable — `356` rows (`33.74%`)
- **Class Balance & Split Strategy:**
  - The dataset has an approximate 2:1 class imbalance towards `NRB`.
  - To prevent data leakage, a **stratified 80/20 train-test split (`random_state=42`)** was performed before fitting any preprocessing steps (`844` rows in training set and `211` rows in `test_data.csv`).
  - A `ColumnTransformer` with `SimpleImputer` and `StandardScaler` was fitted only on the training data and bundled inside each model's scikit-learn `Pipeline`.

---

## c. GitHub Repository Link

- **GitHub Repository Link:** [https://github.com/Infinite-Thinker/BITS_ML_Assignment_2](https://github.com/Infinite-Thinker/BITS_ML_Assignment_2)
- **Live Streamlit App Link:** *(Will be added after deploying on Streamlit Community Cloud)*

### Project Folder Structure
```text
BITS_ML_Assignment_2/
|-- app.py                          # Streamlit web app for evaluating test_data.csv
|-- preprocess.py                   # Stratified 80/20 split and ColumnTransformer setup
|-- train_models.py                 # Script to train all 5 models and compute 6 metrics
|-- eda.py                          # Exploratory data analysis and plots script
|-- requirements.txt                # Required Python libraries with pinned versions
|-- README.md                       # Project documentation and observation tables
|-- test_data.csv                   # 20% holdout test dataset (211 rows, 42 columns)
|-- results.csv                     # Saved 5x6 evaluation metrics table
|-- data/
|   `-- qsar_biodeg.csv             # Full formatted dataset (1,055 rows, 42 columns)
|-- model/
|   |-- logistic_regression.joblib  # Saved Logistic Regression pipeline
|   |-- decision_tree.joblib        # Saved Decision Tree pipeline
|   |-- knn.joblib                  # Saved KNN pipeline
|   |-- naive_bayes.joblib          # Saved Gaussian Naive Bayes pipeline
|   `-- svm.joblib                  # Saved SVM pipeline
`-- assets/
    |-- class_balance.png
    |-- descriptor_scale_disparity.png
    `-- top_descriptor_correlations.png
```

---

## d. Models Used

### Evaluation Metrics Comparison Table (on Holdout Test Set, `211` rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8626** | 0.9136 | 0.8000 | 0.7887 | **0.7943** | **0.6912** |
| **Decision Tree** | 0.7962 | 0.7666 | 0.7059 | 0.6761 | 0.6906 | 0.5391 |
| **KNN** | 0.8483 | 0.9018 | 0.7532 | 0.8169 | 0.7838 | 0.6686 |
| **Naive Bayes** | 0.6682 | 0.8195 | 0.5038 | **0.9437** | 0.6569 | 0.4622 |
| **Support Vector Machine** | 0.8578 | **0.9188** | **0.8060** | 0.7606 | 0.7826 | 0.6777 |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Logistic Regression performed the most consistently across metrics, getting the highest Accuracy (`0.8626`), F1 score (`0.7943`), and MCC (`0.6912`), along with a strong AUC (`0.9136`). Its training accuracy was `0.8898`, which is very close to the test accuracy and shows that the default L2 regularization prevented overfitting even though several molecular descriptors are correlated. |
| **Decision Tree** | The Decision Tree model clearly overfitted the training data. It reached `1.0000` (100%) accuracy on the training set, but dropped down to `0.7962` accuracy and the lowest AUC (`0.7666`) on the test set. Because `max_depth` was left unconstrained, the tree grew deep leaves to memorize noisy combinations of descriptors instead of learning general rules. |
| **KNN** | KNN (`k=7` with distance weighting) gave solid results with `0.8483` Accuracy, `0.9018` AUC, and the second-highest Recall (`0.8169`). Applying `StandardScaler` in the preprocessing pipeline was very important here because features like `C_pct` go up to `100` while others like `Psi_i_1d` are close to `0`, which would have skewed the Euclidean distance calculation without scaling. |
| **Naive Bayes** | Gaussian Naive Bayes had the lowest overall Accuracy (`0.6682`), Precision (`0.5038`), and MCC (`0.4622`), even though its Recall was the highest (`0.9437`). During EDA, we saw 14 pairs of descriptors with high correlation ($|r| > 0.80$). This breaks Naive Bayes' main assumption that all features are independent, causing it to over-predict the positive `RB` class and generate many false positives. |
| **Support Vector Machine** | SVM with the RBF kernel achieved the highest AUC score (`0.9188`) and highest Precision (`0.8060`), while coming in a very close second on Accuracy (`0.8578`) and MCC (`0.6777`). Feature scaling helped the RBF kernel find a clean non-linear boundary between biodegradable and non-biodegradable molecules with minimal false positives. |
| **Overall, Winner for your dataset?** | **Logistic Regression** is the overall winner for this dataset (with **SVM** as a very close runner-up). Logistic Regression leads on 3 out of the 6 metrics—Accuracy (`0.8626`), F1 (`0.7943`), and MCC (`0.6912`)—which is especially important given the 66% vs 34% class imbalance, while also being the fastest model and having the smallest saved file size (`3.3 KB`). |
