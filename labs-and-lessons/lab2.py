def ex1():
    # Import scikit-learn dataset library
    from sklearn import datasets

    # Load dataset
    iris = datasets.load_iris()

    # Creating a DataFrame of given iris dataset.
    import pandas as pd

    data = pd.DataFrame(
        {
            "sepal length": iris.data[:, 0],
            "sepal width": iris.data[:, 1],
            "petal length": iris.data[:, 2],
            "petal width": iris.data[:, 3],
            "species": iris.target,
        }
    )
    iris["target_names"]
    print(data.head())

    # Import train_test_split function
    from sklearn.model_selection import train_test_split

    X = data[["sepal length", "sepal width", "petal length", "petal width"]]  # Features
    y = data["species"]  # Labels

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier

    # Create a Gaussian Classifier
    rf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=rf.predict(X_test)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    # Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # Predict species for a single flower.
    # sepal length = 3, sepal width = 5
    # petal length = 4, petal width = 2
    prediction = rf.predict([[3, 5, 4, 2]])
    # 'setosa', 'versicolor', 'virginica'
    print(prediction)

    # Get numerical feature importances
    importances = list(rf.feature_importances_)

    # Present features and importance scores.
    def showFeatureImportances(importances, feature_list):
        dfImportance = pd.DataFrame()
        for i in range(0, len(importances)):
            dfImportance = dfImportance.append(
                {"importance": importances[i], "feature": feature_list[i]},
                ignore_index=True,
            )

        dfImportance = dfImportance.sort_values(by=["importance"], ascending=False)
        print(dfImportance)

    feature_list = ["sepal length", "sepal width", "petal length", "petal width"]
    showFeatureImportances(importances, feature_list)


def ex2():
    # Import scikit-learn dataset library
    from sklearn import datasets

    # Load dataset
    iris = datasets.load_iris()

    # Creating a DataFrame of given iris dataset.
    import pandas as pd
    import numpy as np

    data = pd.DataFrame(
        {
            "sepal length": iris.data[:, 0],
            "sepal width": iris.data[:, 1],
            "petal length": iris.data[:, 2],
            "petal width": iris.data[:, 3],
            "species": iris.target,
        }
    )
    iris["target_names"]
    print(data.head())

    # Import train_test_split function
    from sklearn.model_selection import train_test_split

    X = data[["sepal length", "sepal width", "petal length", "petal width"]]  # Features
    y = data["species"]  # Labels
    features = np.array(X)
    labels = np.array(y)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

    # Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier

    # Create a Gaussian Classifier
    rf = RandomForestClassifier(n_estimators=100)

    feature_list = list(X.columns)

    # Extract the two most important features
    important_indices = [
        feature_list.index("petal width"),
        feature_list.index("petal length"),
    ]
    train_important = X_train[:, important_indices]
    test_important = X_test[:, important_indices]

    # Train the model using the training sets y_pred=rf.predict(X_test)
    rf.fit(train_important, y_train)

    y_pred = rf.predict(test_important)

    # Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


def ex3():
    # Pandas is used for data manipulation
    import pandas as pd
    from sklearn.metrics import mean_squared_error

    # Read in data and display first 5 rows
    features = pd.read_csv(
        "/Users/elber/Documents/COMP 4948 - Datasets/temperatures.csv"
    )

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print(features)

    # One-hot encode the data using pandas get_dummies
    features = pd.get_dummies(features)

    # Display the first 5 rows of the last 12 columns.
    print(features.head(5))

    # Use numpy to convert to arrays
    import numpy as np

    # Labels are the values we want to predict
    labels = np.array(features["actual"])

    # Remove the labels from the features
    # axis 1 refers to the columns
    features = features.drop("actual", axis=1)

    # Saving feature names for later use
    feature_list = list(features.columns)

    # Convert to numpy array
    features = np.array(features)

    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.25, random_state=42
    )

    # Import the model we are using
    from sklearn.ensemble import RandomForestRegressor

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(
        n_estimators=1600,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features=None,
        max_depth=None,
        bootstrap=True,
    )

    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)

    # Calculate the absolute errors
    errors = abs(predictions - test_labels)

    # Print out the mean absolute error (mae)
    print("Mean Absolute Error:", round(np.mean(errors), 2), "degrees.")

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)

    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)

    print("Accuracy:", round(accuracy, 2), "%.")

    # Print out the mean square error.
    mse = mean_squared_error(test_labels, predictions)
    print("RMSE:", np.sqrt(mse))


def ex4():
    from sklearn import datasets

    iris = datasets.load_iris()

    # Creating a DataFrame of given iris dataset.
    import pandas as pd

    data = pd.DataFrame(
        {
            "sepal length": iris.data[:, 0],
            "sepal width": iris.data[:, 1],
            "petal length": iris.data[:, 2],
            "petal width": iris.data[:, 3],
            "species": iris.target,
        }
    )
    iris["target_names"]
    print(data.head())

    # Import train_test_split function
    from sklearn.model_selection import train_test_split

    X = data[["sepal length", "sepal width", "petal length", "petal width"]]  # Features
    y = data["species"]  # Labels

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    from sklearn.preprocessing import StandardScaler

    sc_x = StandardScaler()
    X_train_scaled = sc_x.fit_transform(X_train)
    X_test_scaled = sc_x.transform(X_test)

    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    def buildModelAndPredict(
        clf, X_train_scaled, X_test_scaled, y_train, y_test, title
    ):
        print("\n**** " + title)
        # Train the model using the training sets y_pred=rf.predict(X_test)
        clf_fit = clf.fit(X_train_scaled, y_train)
        y_pred = clf_fit.predict(X_test_scaled)

        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

        # For explanation see:
        # https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
        print(metrics.classification_report(y_test, y_pred, digits=3))

        # Predict species for a single flower.
        # sepal length = 3, sepal width = 5
        # petal length = 4, petal width = 2
        prediction = clf_fit.predict([[3, 5, 4, 2]])

        # 'setosa', 'versicolor', 'virginica'
        print(prediction)

    lr = LogisticRegression(fit_intercept=True, solver="liblinear")
    buildModelAndPredict(
        lr, X_train_scaled, X_test_scaled, y_train, y_test, "Logistic Regression"
    )

    rf = RandomForestClassifier(n_estimators=200, max_features=3)
    buildModelAndPredict(
        rf, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest"
    )


def main():
    # ex1()
    # ex2()
    # ex3()
    ex4()


if __name__ == "__main__":
    main()
