from src.data.generate_data import generate_iris_dataframe
from src.models.train import train_model


def test_train_model_runs_and_returns_accuracy():
    df = generate_iris_dataframe()
    model, acc, X_test, y_test = train_model(df)
    assert model is not None
    assert 0.0 <= acc <= 1.0
    # basic sanity: predictions length matches y_test
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
