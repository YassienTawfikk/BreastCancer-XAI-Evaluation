from pathlib import Path

# === Ensure base dataset directory exists ===
Path('datasets/').mkdir(parents=True, exist_ok=True)

# === Ensure output directories exist for each model ===
models = ['Logistic Regression', 'Random Forest', 'SVM']
subfolders = ['model', 'plots', 'reports']

for model in models:
    for sub in subfolders:
        Path(f'outputs/{model}/{sub}').mkdir(parents=True, exist_ok=True)
