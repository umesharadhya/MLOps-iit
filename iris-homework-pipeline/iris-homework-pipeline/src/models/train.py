def train_model(data):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd

    # Split the data into features and target
    X = data.drop(columns=['species'])
    y = data['species']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = DecisionTreeClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Return the trained model and test data
    return model, X_test, y_test