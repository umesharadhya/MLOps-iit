from sklearn.datasets import load_iris
import pandas as pd


def load_iris_dataframe() -> pd.DataFrame:
    """Return the Iris dataset as a pandas DataFrame with a `target` column."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df


if __name__ == "__main__":
    df = load_iris_dataframe()
    print(df.head())
