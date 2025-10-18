import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Basic validation: checks required columns and no missing values."""
    required = {"sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "target"}
    if not required.issubset(set(df.columns)):
        return False
    if df[required].isnull().any().any():
        return False
    return True
