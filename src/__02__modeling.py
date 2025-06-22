# src/__02__modeling.py

from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

def run_modeling():
    print("Loading processed data...")
    train_df = pd.read_csv('datasets/processed_train.csv')
    test_df = pd.read_csv('datasets/processed_test.csv')

    x_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    x_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Define output base path
    output_base = Path('outputs')

    # Define models
    models_dict = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }

    for name, model in models_dict.items():
        print(f"\nTraining {name}...")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        # Paths
        model_dir = output_base / name
        model_path = model_dir / 'model' / f"{name.lower().replace(' ', '_')}_model.pkl"
        report_path = model_dir / 'reports' / 'classification_report.txt'
        cm_path = model_dir / 'plots' / 'confusion_matrix.png'
        roc_path = model_dir / 'plots' / 'roc_curve.png'
        prc_path = model_dir / 'plots' / 'precision_recall_curve.png'

        # Save model
        joblib.dump(model, model_path)

        # Save classification report
        report = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'])
        with open(report_path, 'w') as f:
            f.write(report)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()

        # Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {name}")
        plt.savefig(prc_path, bbox_inches='tight')
        plt.close()

        print(f"{name} results saved in outputs/{name}/")

