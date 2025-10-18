from src.data.loader import load_iris_dataframe
from src.models.trainer import train, save, load
from pathlib import Path


def test_model_save_and_load(tmp_path):
    df = load_iris_dataframe()
    model, acc, X_test, y_test = train(df)
    path = tmp_path / "model.joblib"
    save(model, path)
    loaded = load(path)
    assert loaded is not None
    preds = loaded.predict(X_test)
    assert len(preds) == len(y_test)
