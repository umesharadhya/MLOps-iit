from src.data.loader import load_iris_dataframe
from src.models.trainer import train


def test_train_model_runs_and_returns_accuracy():
    df = load_iris_dataframe()
    model, acc, X_test, y_test = train(df)
    assert model is not None
    assert 0.0 <= acc <= 1.0
    # basic sanity: predictions length matches y_test
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
