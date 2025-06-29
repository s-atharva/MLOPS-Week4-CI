import pytest
from src.preprocess import load_data
from src.model import train_model

def test_model_accuracy_above_threshold():
    df = load_data()
    accuracy = train_model(df)
    assert accuracy > 0.80, f"Model accuracy is low: {accuracy:.2f}"