import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Default paths
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
RESULTS_TABLES = os.path.join(BASE_DIR, "results", "tables")
RESULTS_FIGURES = os.path.join(BASE_DIR, "results", "figures")

os.makedirs(RESULTS_TABLES, exist_ok=True)
os.makedirs(RESULTS_FIGURES, exist_ok=True)


def evaluate_models(test_path: str = TEST_PATH,
                    models_dir: str = MODELS_DIR) -> dict:
    """
    Evaluate trained models on the processed test dataset.

    Steps:
    1. Load processed test dataset.
    2. Load each trained model from disk.
    3. Generate predictions and compute metrics.
    4. Save evaluation results into results/tables and confusion matrices into results/figures.

    Parameters
    ----------
    test_path : str, optional
        Absolute path to the processed test dataset.
    models_dir : str, optional
        Absolute path to the directory containing trained model pickle files.

    Returns
    -------
    results : dict
        Dictionary mapping model names to their evaluation metrics.
    """
    # 1. Load test dataset
    df = pd.read_csv(test_path)
    if "Transported" not in df.columns:
        raise ValueError("Processed test dataset must include 'Transported' as the target column.")
    X_test = df.drop("Transported", axis=1)
    y_test = df["Transported"]

    results = {}

    # 2. Loop through models
    for file in os.listdir(models_dir):
        if file.endswith(".pkl"):
            model_name = file.replace(".pkl", "")
            model_path = os.path.join(models_dir, file)

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            try:
                # 3. Predict
                y_pred = model.predict(X_test)

                # 4. Metrics
                acc = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                results[model_name] = {
                    "accuracy": acc,
                    "report": report
                }

                # Save metrics table
                metrics_df = pd.DataFrame(report).transpose()
                metrics_path = os.path.join(RESULTS_TABLES, f"{model_name}_metrics.csv")
                metrics_df.to_csv(metrics_path)

                # Save confusion matrix plot
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=[0, 1], yticklabels=[0, 1])
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title(f"Confusion Matrix - {model_name}")
                fig_path = os.path.join(RESULTS_FIGURES, f"{model_name}_cm.png")
                plt.savefig(fig_path)
                plt.close()

                print(f"{model_name}: accuracy={acc:.4f} (results saved)")

            except Exception as e:
                print(f"Evaluation failed for {model_name}: {e}")
                results[model_name] = None

    return results


if __name__ == "__main__":
    evaluate_models()
