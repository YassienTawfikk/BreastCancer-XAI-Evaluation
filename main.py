import src
from src import __01__preprocessing as preprocess
from src import __02__modeling as modeling
from src import __03__explainability as explainability


def main():
    print("Running preprocessing step...")
    preprocess.run_preprocessing()

    print("Running modeling step...")
    modeling.run_modeling()

    print("Running explainability step...")
    explainability.run_explainability()

    print("All steps completed successfully.")


if __name__ == "__main__":
    main()
