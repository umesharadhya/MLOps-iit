import pandas as pd
from src.data.loader import load_iris_dataframe
from src.validation.validator import validate_dataframe


def test_generate_dataframe_has_expected_columns():
    df = load_iris_dataframe()
    assert "sepal length (cm)" in df.columns
    assert "target" in df.columns


def test_validate_dataframe_success():
    df = load_iris_dataframe()
    assert validate_dataframe(df) is True

