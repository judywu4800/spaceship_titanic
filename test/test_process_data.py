import os
import pytest
import pandas as pd
from src.pipeline.process_data import process_data, BASE_DIR

def test_process_data_train():
    """Test that process_data works on train.csv and produces expected output."""
    input_path = os.path.join(BASE_DIR, "data", "raw", "train.csv")
    output_path = os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")

    df = process_data(input_path=input_path, output_path=output_path)

    # 1. Check file exists
    assert os.path.exists(output_path), "Processed train file not created"

    # 2. Check dataframe returned
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "Processed train dataframe is empty"

    # 3. Check target column exists
    assert "Transported" in df.columns, "Processed train missing target column"

    # 4. Check no missing values in critical columns
    assert df["Transported"].isna().sum() == 0, "Transported has missing values"


