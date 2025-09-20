import os
from src.pipeline.process_data import process_data
from src.analysis.fit_models import fit_models
from src.analysis.evaluate import evaluate_models

# Project root
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    print("=== Step 1: Processing data ===")
    train_df = process_data(
        input_path=os.path.join(BASE_DIR, "data", "raw", "train.csv"),
        output_path=os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")
    )
    test_df = process_data(
        input_path=os.path.join(BASE_DIR, "data", "raw", "test.csv"),
        output_path=os.path.join(BASE_DIR, "data", "processed", "test_processed.csv")
    )

    print("\n=== Step 2: Training models ===")
    results = fit_models()

    print("\n=== Step 3: Evaluating models ===")
    # For assignment reproducibility, evaluate on train_processed (with labels)
    eval_results = evaluate_models(
        test_path=os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")
    )

    print("\nPipeline finished successfully.")
