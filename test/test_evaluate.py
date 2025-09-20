import os
import pytest
import pandas as pd
from src.analysis.evaluate import evaluate_models, BASE_DIR
from src.analysis.fit_models import MODELS

def test_evaluate_models_output():
    """Test that evaluate_models produces expected results and output files."""
    results = evaluate_models(
        test_path=os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")
    )

    # 1. Check results structure
    assert isinstance(results, dict)
    for name in MODELS.keys():
        assert name in results

    # 2. Check metrics tables are saved
    tables_dir = os.path.join(BASE_DIR, "results", "tables")
    for name in MODELS.keys():
        metrics_file = os.path.join(tables_dir, f"{name}_metrics.csv")
        assert os.path.exists(metrics_file), f"Metrics file missing: {metrics_file}"
        df = pd.read_csv(metrics_file)
        assert "precision" in df.columns or "f1-score" in df.columns

    # 3. Check confusion matrices are saved
    figs_dir = os.path.join(BASE_DIR, "results", "figures")
    for name in MODELS.keys():
        fig_file = os.path.join(figs_dir, f"{name}_cm.png")
        assert os.path.exists(fig_file), f"Confusion matrix missing: {fig_file}"
