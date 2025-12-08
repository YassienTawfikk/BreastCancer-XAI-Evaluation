import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from jsonargparse import CLI
from breast_cancer.modeling import BreastCancerModel, ModelConfig
from breast_cancer import preprocessing, explainability

class TrainingCLI:
    def fit(
        self,
        data: dict = {
            "ndim": 3,  # Placeholder to match user request template, though likely unused by current simple logic
            "batch_size": 1,
            "num_workers": 12
        },
        model: ModelConfig = ModelConfig(),
        trainer: dict = {
            "max_epochs": 1000,
            "accelerator": "gpu"
        }
    ):
        """
        Run the full training pipeline.
        
        Args:
            data: Data configuration (placeholders for now to match template).
            model: Model hyperparameters.
            trainer: Trainer configuration (placeholders for now).
        """
        print(f"Starting pipeline with config:")
        print(f"  Model: {model}")
        print(f"  Data: {data}")
        print(f"  Trainer: {trainer}")

        # 1. Preprocessing
        print("\n--- [Step 1/3] Preprocessing ---")
        preprocessing.run_preprocessing()
        
        # 2. Modeling
        print("\n--- [Step 2/3] Modeling ---")
        bc_model = BreastCancerModel(config=model)
        bc_model.run()
        
        # 3. Explainability (Optional default run)
        print("\n--- [Step 3/3] Explainability ---")
        explainability.run_explainability(index=24) 

if __name__ == "__main__":
    CLI(TrainingCLI)
