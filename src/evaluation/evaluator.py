from sklearn.metrics import accuracy_score
import pandas as pd


def evaluate_model(model, X, y):
    preds = model.predict(X)
    return {"accuracy": float(accuracy_score(y, preds))}
