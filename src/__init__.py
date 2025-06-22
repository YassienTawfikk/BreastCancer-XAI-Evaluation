from pathlib import Path

processed_dataset_path = '../datasets/'

# Ensure directory exists
folder_path = Path(processed_dataset_path)
folder_path.mkdir(parents=True, exist_ok=True)

# === Define base output directory ===
output_base = Path('../outputs')

# === Define subdirectories for each model ===
models = ['Logistic Regression', 'Random Forest', 'SVM']
subfolders = ['model', 'plots', 'reports']

# === Create full directory tree ===
for model in models:
    for sub in subfolders:
        path = output_base / model / sub
        path.mkdir(parents=True, exist_ok=True)
