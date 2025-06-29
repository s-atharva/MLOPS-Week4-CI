import pytest
from src.preprocess import load_data

def test_column_count():
    df = load_data()
    assert df.shape[1] == 5, "Dataset should have 5 columns (4 features + 1 target)."

def test_no_missing_values():
    df = load_data()
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values."
