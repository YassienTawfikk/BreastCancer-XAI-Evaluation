# ===============================
# 📦 Imports
# ===============================
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from pathlib import Path
import warnings

# ===============================
# ⚠️ Ignore Warnings
# ===============================
warnings.filterwarnings("ignore", category=FutureWarning)

# ===============================
# 📥 Load Dataset
# ===============================
data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='label')

# ===============================
# ✂️ Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

# ===============================
# 🔍 Feature Scaling
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 🔄 Apply SMOTE
# ===============================
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# ===============================
# 💾 Save Processed Training Data
# ===============================
processed_dataset_path = '../datasets/processed'  # ✅ corrected folder name

# Ensure directory exists
folder_path = Path(processed_dataset_path)
folder_path.mkdir(parents=True, exist_ok=True)

# Create DataFrame and save to CSV
processed_train = pd.DataFrame(X_train_smote, columns=data.feature_names)
processed_train['target'] = y_train_smote
processed_train.to_csv(f"{processed_dataset_path}/processed_train.csv", index=False)
