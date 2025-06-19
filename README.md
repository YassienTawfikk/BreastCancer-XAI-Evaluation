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

## Evaluation Summary _(To be completed after training)_

| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression |          |           |        |          |         |
| Random Forest       |          |           |        |          |         |
| SVM                 |          |           |        |          |         |

---

## Explainability Output (SHAP)

> [Insert visual here after running SHAP summary plot]  
> Example: ![SHAP Summary Plot](./shap_summary.png)

---

## File Structure

```
BreastCancer-XAI-Evaluation/
│
├── data/                      # Folder for datasets (raw or processed)
│   └── breast_cancer.csv
│
├── notebooks/
│   └── tumor_classifier_workflow.ipynb   # Full Jupyter workflow (EDA to explainability)
│
├── src/                       # Python scripts for modular code
│   ├── __init__.py
│   ├── preprocessing.py       # Functions for cleaning, SMOTE, scaling
│   ├── modeling.py            # Train/test split, model training, metrics
│   └── explainability.py      # SHAP or LIME interpretability
│
├── outputs/
│   ├── models/                # Saved trained models (e.g., .pkl or .joblib)
│   ├── plots/                 # Visualizations, ROC curves, SHAP plots
│   └── reports/               # Clinical-style report / results summary
│
├── .gitignore                 # All ignored files
├── requirements.txt           # All needed Python packages
├── README.md                  # Project summary, setup, usage
└── main.py                    # Optional: script to run everything end-to-end
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
