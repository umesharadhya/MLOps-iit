from src.data.generate_data import generate_iris_dataframe
from src.models.train import train_model, save_model
from pathlib import Path


def main():
    df = generate_iris_dataframe()
    model, acc, X_test, y_test = train_model(df)
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    model_path = out / "rf_model.joblib"
    save_model(model, model_path)
    print(f"Trained model saved to {model_path} with accuracy={acc:.3f}")


if __name__ == "__main__":
    main()
