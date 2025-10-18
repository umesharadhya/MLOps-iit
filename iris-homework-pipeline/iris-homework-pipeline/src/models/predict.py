def make_predictions(model_path, new_data):
    import joblib
    import pandas as pd

    # Load the trained model
    model = joblib.load(model_path)

    # Make predictions on the new data
    predictions = model.predict(new_data)

    return predictions