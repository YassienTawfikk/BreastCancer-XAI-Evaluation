import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path


def run_explainability(index: int = 0) -> None:
    """Generate SHAP summary and local explanation plots for all models."""
    print("Loading processed data and models for explainability...")

    train_df = pd.read_csv('datasets/processed_train.csv')
    test_df = pd.read_csv('datasets/processed_test.csv')

    X_train = train_df.drop('target', axis=1)
    X_test = test_df.drop('target', axis=1)

    base = Path('outputs')
    models = {
        'Logistic Regression': joblib.load(base / 'Logistic Regression' / 'model' / 'logistic_regression_model.pkl'),
        'Random Forest': joblib.load(base / 'Random Forest' / 'model' / 'random_forest_model.pkl'),
        'SVM': joblib.load(base / 'SVM' / 'model' / 'svm_model.pkl'),
    }

    for name, model in models.items():
        print(f'Explaining {name}...')
        if name == 'SVM':
            background = shap.sample(X_train, 50, random_state=42)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_test.iloc[:50])
            values = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]
            sv = shap.Explanation(
                values=values,
                base_values=explainer.expected_value[1],
                data=X_test.iloc[:50].values,
                feature_names=X_test.columns,
            )
        else:
            explainer = shap.Explainer(model, X_train)
            if hasattr(explainer, 'model') and hasattr(explainer.model, 'output_type') and explainer.model.output_type == 'probability':
                shap_values = explainer(X_test, check_additivity=False)
            else:
                shap_values = explainer(X_test)
            if len(shap_values.shape) == 3:
                sv = shap.Explanation(
                    values=shap_values.values[:, :, 1],
                    base_values=shap_values.base_values[:, 1],
                    data=shap_values.data,
                    feature_names=shap_values.feature_names,
                )
            else:
                sv = shap_values

        plot_dir = base / name / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)

        summary_path = plot_dir / 'shap_summary.png'
        features = X_test.iloc[:sv.shape[0]] if not isinstance(sv, shap._explanation.Explanation) else X_test.iloc[:len(sv)]
        shap.summary_plot(sv, features=features, show=False)
        plt.tight_layout()
        plt.savefig(summary_path, bbox_inches='tight')
        plt.close()

        local_path = plot_dir / f'shap_local_{index}.png'
        if isinstance(sv, shap._explanation.Explanation):
            shap.plots.waterfall(sv[min(index, len(sv)-1)], show=False)
        else:
            exp = shap.Explanation(
                values=sv[min(index, sv.shape[0]-1)],
                base_values=explainer.expected_value[1] if name == 'SVM' else explainer.expected_value,
                data=X_test.iloc[min(index, len(X_test)-1)].values,
                feature_names=X_test.columns,
            )
            shap.plots.waterfall(exp, show=False)
        plt.savefig(local_path, bbox_inches='tight')
        plt.close()

        print(f'SHAP plots saved to {plot_dir}/')
