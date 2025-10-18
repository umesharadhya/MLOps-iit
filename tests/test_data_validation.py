import pandas as pd
from src.data.generate_data import generate_iris_dataframe
from src.validation.validator import validate_dataframe


def test_generate_dataframe_has_expected_columns():
    df = generate_iris_dataframe()
    assert "sepal length (cm)" in df.columns
    assert "target" in df.columns


def test_validate_dataframe_success():
    df = generate_iris_dataframe()
    assert validate_dataframe(df) is True

