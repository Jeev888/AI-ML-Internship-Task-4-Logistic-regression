# AI & ML INTERNSHIP - Task 4: Classification with Logistic Regression

This repository contains the solution for Task 4 of the AI & ML internship, which focuses on building a binary classifier using the Breast Cancer Wisconsin (Diagnostic) Data Set. This task demonstrates the end-to-end process of a classification problem, from data preparation to model evaluation and interpretation.

---

## 1. Objective

The objective of this task was to implement and understand logistic regression for binary classification. This includes:
* **Preprocessing a binary classification dataset.**
* **Splitting and standardizing features.**
* **Training and evaluating a Logistic Regression model.**
* **Interpreting a confusion matrix, precision, and recall.**
* **Understanding and tuning the classification threshold.**
* **Visualizing the sigmoid function and the ROC-AUC curve.**

---

## 2. Dataset

* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
* **Source:** The dataset was downloaded programmatically using `kagglehub`.
* **File:** `data.csv`

---

## 3. Tools & Libraries

* **Pandas:** For data manipulation, loading, and preprocessing.
* **NumPy:** For numerical operations.
* **Scikit-learn:** The primary tool for the machine learning workflow, including data splitting (`train_test_split`), feature scaling (`StandardScaler`), model training (`LogisticRegression`), and evaluation metrics (`confusion_matrix`, `precision_score`, `recall_score`, `roc_auc_score`).
* **Matplotlib & Seaborn:** For creating visualizations.

---

## 4. Project Workflow

The following steps were performed to build and evaluate the logistic regression model.

### 4.1. Data Preprocessing and Standardization

* **Initial Data Check:** The dataset was loaded, and an initial check confirmed the absence of missing values in all relevant columns. The `id` and `Unnamed: 32` columns were identified as irrelevant and dropped.
* **Label Encoding:** The categorical `diagnosis` column ('M' and 'B') was converted to numerical format (`1` and `0`) to serve as the target variable for the model.
* **Feature Standardization:** All features were standardized using `StandardScaler`. This is a critical step for logistic regression to ensure all features are on a similar scale, leading to more efficient and stable model training.

### 4.2. Model Training and Evaluation (Default Threshold)

* A **Logistic Regression model** was trained on the scaled training data.
* The model was evaluated using a **default threshold of 0.5** to classify predictions.
* **Confusion Matrix:** The confusion matrix showed that the model made only a few incorrect predictions.
    ```
    [[70  1]
     [ 2 41]]
    ```
    * **Inference:** The model correctly predicted 70 benign cases (True Negatives) and 41 malignant cases (True Positives). It had 1 False Positive and 2 False Negatives.

### 4.3. Threshold Tuning for Improved Recall

* **Rationale:** In a medical context, a **False Negative** (missing a malignant diagnosis) is a more severe error than a False Positive (a false alarm). To prioritize catching as many actual malignant cases as possible, the classification threshold was lowered from the default of 0.5 to **0.3**.
* **Re-evaluated Metrics:** The model was re-evaluated with the new threshold, and a new confusion matrix was generated.
    ```
    [[67  4]
     [ 1 42]]
    ```
    * **Inference:** By lowering the threshold, the model successfully **caught one of the previously missed malignant cases** (fewer False Negatives). However, this came at the cost of increasing the number of False Positives from 1 to 4. This trade-off between Precision and Recall is a key concept in classification model tuning.

### 4.4. Visualization

The notebook includes key visualizations to understand the model and its performance:
* **Sigmoid Function:** A conceptual plot of the sigmoid function visually demonstrates how logistic regression maps any value to a probability between 0 and 1.
* **ROC-AUC Curve:** The ROC (Receiver Operating Characteristic) curve was plotted, showing the model's performance across all possible thresholds. The **Area Under the Curve (AUC) score was 1.00**, indicating that the model is nearly perfect at distinguishing between the two classes.

<br>
<br>

---

## 5. Conclusion

The logistic regression model built for this task is highly effective, achieving a near-perfect ROC-AUC score. The process demonstrates a strong understanding of binary classification, including the importance of data preprocessing, standardization, and the practical implications of tuning the classification threshold in real-world applications.
