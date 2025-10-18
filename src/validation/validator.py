import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Basic validation: checks required columns and no missing values."""
    required = {"sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "target"}
    if not required.issubset(set(df.columns)):
        return False
    # pandas does not accept sets as column indexers; convert to list
    required_list = list(required)
    if df[required_list].isnull().any().any():
        return False
    return True
