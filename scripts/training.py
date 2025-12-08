import argparse
import sys
from pathlib import Path

# Add project root to sys.path so we can import breast_cancer module
# This assumes scripts/training.py is one level deep (parent is project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from breast_cancer import preprocessing, modeling, explainability

class CLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Breast Cancer XAI Evaluation Pipeline")
        self._add_arguments()
        
    def _add_arguments(self):
        self.parser.add_argument(
            '--steps', 
            nargs='+', 
            default=['all'], 
            choices=['preprocessing', 'modeling', 'explainability', 'all'],
            help="Pipeline steps to execute. Default is 'all'."
        )
        self.parser.add_argument(
            '--explain-index',
            type=int,
            default=24,
            help="Index of the test sample to explain (for local explainability plots). Default is 24."
        )

    def run(self):
        args = self.parser.parse_args()
        steps = args.steps
        
        if 'all' in steps:
            steps = ['preprocessing', 'modeling', 'explainability']

        print(f"Executing steps: {', '.join(steps)}")

        if 'preprocessing' in steps:
            print("\n--- [Step 1/3] Preprocessing ---")
            preprocessing.run_preprocessing()
        
        if 'modeling' in steps:
            print("\n--- [Step 2/3] Modeling ---")
            modeling.run_modeling()
            
        if 'explainability' in steps:
            print("\n--- [Step 3/3] Explainability ---")
            explainability.run_explainability(index=args.explain_index)

        print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    cli = CLI()
    cli.run()
