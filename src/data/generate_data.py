from sklearn.datasets import load_iris
import pandas as pd


def generate_iris_dataframe():
    """Return iris dataset as a pandas DataFrame with target."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df


if __name__ == "__main__":
    df = generate_iris_dataframe()
    print(df.head())
