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

| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------------------|----------|-----------|--------|----------|---------|
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
├── Tumor\_Prediction\_Evaluation.ipynb     # Full pipeline notebook
├── README.md                             # This file
├── shap\_summary.png                      # SHAP image (optional)
├── requirements.txt                      # (optional)

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
