# BreastCancer-XAI-Evaluation

> Exploring explainable AI for breast cancer classification through SHAP insights, precision-recall metrics, and imbalanced data handling.

---

## Project Overview

This project builds and evaluates a binary machine learning model to classify breast tumors as **benign or malignant** using clinical diagnostic features.  
The main focus is not just accuracy, but on **reliable evaluation** and **explainability**, two critical aspects in medical AI applications.

---

## Objectives

- Build baseline ML models (Logistic Regression, Random Forest, SVM)
- Apply clinical-style evaluation: **Precision, Recall, F1-score, ROC-AUC, PR curves**
- Address **class imbalance** using SMOTE
- Use **SHAP** to interpret and visualize model predictions
- Emphasize model **transparency and trustworthiness**

---

## Dataset

**Breast Cancer Wisconsin Diagnostic Dataset**  
Source: `scikit-learn.datasets.load_breast_cancer()`

- 30 numerical features (e.g., radius, concavity, texture)
- Target: `0 = benign`, `1 = malignant`

---

## Tools & Technologies

- **Language**: Python
- **Libraries**: scikit-learn, pandas, matplotlib, seaborn, imbalanced-learn, SHAP
- **IDE**: Jupyter Notebook

---

Sure! Here’s a complete, well-structured replacement for your `## Evaluation Summary` section in the `README.md` — explaining all reported metrics (Accuracy, Precision, Recall, F1, AUC) and why they’re high:

---

## Evaluation Summary

| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| SVM                 | 0.96     | 0.97      | 0.96   | 0.97     | 0.99    |
| Random Forest       | 0.95     | 0.95      | 0.95   | 0.95     | 0.99    |
| Logistic Regression | 0.96     | 0.96      | 0.96   | 0.96     | 0.99    |

### Metric Interpretation & Why They're High

The consistently high scores across all models can be attributed to the characteristics of the dataset and the preprocessing pipeline:

* **Accuracy (95–96%)**: This reflects the overall proportion of correctly classified tumors. The dataset used is well-balanced (after SMOTE), and the features offer strong separation between benign and malignant classes — leading to high correct classification rates.

* **Precision (95–97%)**: High precision means that when the model predicts a tumor as malignant, it's usually correct. This is especially strong in SVM and Logistic Regression, which learned well-defined boundaries thanks to clean feature distributions and scaling.

* **Recall (95–96%)**: Also known as sensitivity, this indicates how many actual malignant tumors were correctly identified. The model performance here suggests strong coverage, with few false negatives — crucial in a cancer detection task.

* **F1 Score (95–97%)**: The harmonic mean of precision and recall. These high values show that the models are balanced — minimizing both false positives and false negatives.

* **ROC AUC (0.99)**: Area Under the ROC Curve quantifies how well the model ranks malignant cases higher than benign ones. The near-perfect AUC values reflect that the dataset has **clear and separable patterns**, and that the classifiers were effective in capturing those without overfitting.

---

### Note on Dataset Limitations

The dataset used (`sklearn.datasets.load_breast_cancer`) is a **clean, pre-engineered academic dataset**. It does not include real-world noise, imaging artifacts, or mislabeled data. While it is excellent for benchmarking algorithms, such high scores may not directly translate to real clinical environments without further validation on more complex datasets.

---


## Explainability Output (SHAP)

> [Insert visual here after running SHAP summary plot]  
> Example: ![SHAP Summary Plot](./shap_summary.png)

---

## File Structure

```
BreastCancer-XAI-Evaluation/
│
├── datasets/                             # Folder for datasets (raw or processed)
│   └── breast_cancer.csv
│
├── notebooks/
│   ├── 01_preprocessing.ipynb            # Functions for cleaning, SMOTE, scaling
│   ├── 02_modeling.ipynb                 # Train/test split, model training, metrics
│   └── 03_explainability.ipynb           # Full Jupyter workflow (EDA to explainability)
│
├── src/                                  # Python scripts for modular code
│   ├── __init__.py
│   ├── __01__preprocessing.py            # Functions for cleaning, SMOTE, scaling
│   ├── __02__modeling.py                 # Train/test split, model training, metrics
│   └── __03__explainability.py           # SHAP or LIME interpretability
│
├── outputs/
│   ├── Logistic Regression/
│   │   ├── model/                        # Saved trained models (e.g., .pkl or .joblib)  
│   │   ├── plots/                        # Visualizations, ROC curves, SHAP plots
│   │   └── report/                       # Report AUC, ROC, Accuracy
│   ├── SVM/
│   │   └── .../
│   └── Random Forest/
│       └── .../
│
├── .gitignore                            # All ignored files
├── requirements.txt                      # All needed Python packages
├── README.md                             # Project summary, setup, usage
└── main.py                               # Optional: script to run everything end-to-end
```

---

## Key Learnings

- Accuracy isn’t enough: **recall and precision matter** in cancer diagnosis
- SHAP explains why predictions were made — **not just what**
- Handling imbalance significantly improves model reliability

---

## Author

<div>
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/YassienTawfikk" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126521373?v=4" width="150px;" alt="Yassien Tawfik"/>
        <br>
        <sub><b>Yassien Tawfik</b></sub>
      </a>
    </td>
  </tr>
</table>
</div>
