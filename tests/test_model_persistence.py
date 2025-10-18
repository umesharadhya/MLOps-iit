from src.data.generate_data import generate_iris_dataframe
from src.models.train import train_model, save_model, load_model
from pathlib import Path


def test_model_save_and_load(tmp_path):
    df = generate_iris_dataframe()
    model, acc, X_test, y_test = train_model(df)
    path = tmp_path / "model.joblib"
    save_model(model, path)
    loaded = load_model(path)
    assert loaded is not None
    preds = loaded.predict(X_test)
    assert len(preds) == len(y_test)
