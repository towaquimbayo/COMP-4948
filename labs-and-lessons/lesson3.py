def ex1():
    import pandas as pd

    # Get the housing data
    df = pd.read_csv(
        "/Users/elber/Documents/COMP 4948 - Datasets/housing_classification.csv"
    )
    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head(5))

    # Split into two sets
    y = df["price"]
    X = df.drop("price", axis=1)

    from sklearn.ensemble import BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import RidgeClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report

    # Create classifiers
    knn = KNeighborsClassifier()
    svc = SVC()
    rg = RidgeClassifier()

    # Build array of classifiers.
    classifierArray = [knn, svc, rg]

    def showStats(classifier, scores):
        print(classifier + ":    ", end="")
        strMean = str(round(scores.mean(), 2))

        strStd = str(round(scores.std(), 2))
        print("Mean: " + strMean + "   ", end="")
        print("Std: " + strStd)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    def evaluateModel(model, X_test, y_test, title):
        print("\n*** " + title + " ***")
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions)
        print(report)

    # Search for the best classifier.
    for clf in classifierArray:
        modelType = clf.__class__.__name__

        # Create and evaluate stand-alone model.
        clfModel = clf.fit(X_train, y_train)
        evaluateModel(clfModel, X_test, y_test, modelType)

        # max_features means the maximum number of features to draw from X.
        # max_samples sets the percentage of available data used for fitting.
        bagging_clf = BaggingClassifier(
            clf, max_samples=0.4, max_features=3, n_estimators=100
        )
        baggedModel = bagging_clf.fit(X_train, y_train)
        evaluateModel(baggedModel, X_test, y_test, "Bagged: " + modelType)


def ex2():
    import pandas as pd

    # Get the housing data
    df = pd.read_csv(
        "/Users/elber/Documents/COMP 4948 - Datasets/housing_classification.csv"
    )
    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head(5))

    # Split into two sets
    y = df["price"]
    X = df.drop("price", axis=1)

    from sklearn.ensemble import BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import RidgeClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression

    # Create classifiers
    knn = KNeighborsClassifier()
    svc = SVC()
    rg = RidgeClassifier()
    lr = LogisticRegression(fit_intercept=True, solver="liblinear")

    # Build array of classifiers.
    classifierArray = [knn, svc, rg, lr]

    def showStats(classifier, scores):
        print(classifier + ":    ", end="")
        strMean = str(round(scores.mean(), 2))

        strStd = str(round(scores.std(), 2))
        print("Mean: " + strMean + "   ", end="")
        print("Std: " + strStd)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    def evaluateModel(model, X_test, y_test, title):
        print("\n*** " + title + " ***")
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions)
        print(report)

    # Search for the best classifier.
    for clf in classifierArray:
        modelType = clf.__class__.__name__

        # Create and evaluate stand-alone model.
        clfModel = clf.fit(X_train, y_train)
        evaluateModel(clfModel, X_test, y_test, modelType)

        # max_features means the maximum number of features to draw from X.
        # max_samples sets the percentage of available data used for fitting.
        bagging_clf = BaggingClassifier(
            clf, max_samples=0.4, max_features=3, n_estimators=100
        )
        baggedModel = bagging_clf.fit(X_train, y_train)
        evaluateModel(baggedModel, X_test, y_test, "Bagged: " + modelType)


def ex3():
    import pandas as pd
    from sklearn.ensemble import BaggingRegressor
    from sklearn.linear_model import LinearRegression

    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.metrics import mean_squared_error

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Load and prepare data.
    FOLDER = "/Users/elber/Documents/COMP 4948 - Datasets/"
    FILE = "petrol_consumption.csv"
    dataset = pd.read_csv(FOLDER + FILE)
    print(dataset)
    X = dataset.copy()
    del X["Petrol_Consumption"]
    y = dataset[["Petrol_Consumption"]]

    feature_combo_list = []

    def evaluateModel(model, X_test, y_test, title, num_estimators, max_features):
        print("\n****** " + title)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Store statistics and add to list.
        stats = {
            "type": title,
            "rmse": rmse,
            "estimators": num_estimators,
            "features": max_features,
        }
        feature_combo_list.append(stats)

    num_estimator_list = [750, 800, 900, 1000]
    max_features_list = [0.2, 0.4, 0.6]
    max_samples_list = [0.25, 0.5, 0.75, 1.0]

    for num_estimators in num_estimator_list:
        for max_features in max_features_list:
            for max_samples in max_samples_list:
                # Create random split.
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                import numpy as np

                # Build linear regression ensemble.
                ensembleModel = BaggingRegressor(
                    estimator=LinearRegression(),
                    max_features=max_features,
                    # Can be percent (float) or actual
                    # total of samples (int).
                    max_samples=max_samples,
                    n_estimators=num_estimators,
                ).fit(X_train, y_train.values.ravel())
                evaluateModel(
                    ensembleModel,
                    X_test,
                    y_test,
                    "Ensemble",
                    num_estimators,
                    max_features,
                )

                # Build stand alone linear regression model.
                model = LinearRegression()
                model.fit(X_train, y_train)
                evaluateModel(model, X_test, y_test, "Linear Regression", None, None)

    # Build data frame with dictionary objects.
    dfStats = pd.DataFrame()
    print(dfStats)
    for combo in feature_combo_list:
        dfStats = pd.concat(
            [dfStats, pd.DataFrame.from_records([combo])], ignore_index=True
        )

    # Sort and show all combinations.
    # Show all rows
    pd.set_option("display.max_rows", None)
    dfStats = dfStats.sort_values(by=["type", "rmse"])
    print(dfStats)


def ex5():
    import pandas as pd
    from sklearn.metrics import classification_report

    # Get the housing data
    df = pd.read_csv(
        "/Users/elber/Documents/COMP 4948 - Datasets/housing_classification.csv"
    )
    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head(5))

    # Split into two sets
    y = df["price"]
    X = df.drop("price", axis=1)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    from sklearn.model_selection import cross_val_score
    from mlxtend.classifier import EnsembleVoteClassifier
    from xgboost import XGBClassifier, plot_importance
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    ada_boost = AdaBoostClassifier()
    grad_boost = GradientBoostingClassifier()
    xgb_boost = XGBClassifier()
    lr = LogisticRegression(fit_intercept=True, solver="liblinear")
    classifiers = [ada_boost, grad_boost, xgb_boost, lr]

    for clf in classifiers:
        print(clf.__class__.__name__)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        report = classification_report(y_test, predictions)
        print(report)


def ex6():
    import pandas as pd
    from sklearn.metrics import classification_report

    # Get the housing data
    df = pd.read_csv("/Users/elber/Documents/COMP 4948 - Datasets/iris_v2.csv")
    dict_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    df["target"] = df["iris_type"].map(dict_map)

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head(5))

    # Split into two sets
    y = df["target"]
    X = df.copy()
    del X["target"]
    del X["iris_type"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    from sklearn.model_selection import cross_val_score
    from mlxtend.classifier import EnsembleVoteClassifier
    from xgboost import XGBClassifier, plot_importance
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

    ada_boost = AdaBoostClassifier()
    grad_boost = GradientBoostingClassifier()
    xgb_boost = XGBClassifier()
    classifiers = [ada_boost, grad_boost, xgb_boost]

    for clf in classifiers:
        print(clf.__class__.__name__)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        report = classification_report(y_test, predictions)
        print(report)


def ex8():
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Prep data.
    PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
    CSV_DATA = "USA_Housing.csv"
    dataset = pd.read_csv(PATH + CSV_DATA)
    print(dataset.head())
    X = dataset[
        [
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population",
        ]
    ].values
    y = dataset["Price"]

    def getUnfitModels():
        models = list()
        models.append(ElasticNet())
        models.append(SVR(gamma="scale"))
        models.append(DecisionTreeRegressor())
        models.append(AdaBoostRegressor())
        models.append(RandomForestRegressor(n_estimators=200))
        models.append(ExtraTreesRegressor(n_estimators=200))
        return models

    def evaluateModel(y_test, predictions, model):
        mse = mean_squared_error(y_test, predictions)
        rmse = round(np.sqrt(mse), 3)
        print(" RMSE:" + str(rmse) + " " + model.__class__.__name__)

    def fitBaseModels(X_train, y_train, X_test, models):
        dfPredictions = pd.DataFrame()

        # Fit base model and store its predictions in dataframe.
        for i in range(0, len(models)):
            models[i].fit(X_train, y_train)
            predictions = models[i].predict(X_test)
            colName = str(i)
            # Add base model predictions to column of data frame.
            dfPredictions[colName] = predictions
        return dfPredictions, models

    def fitStackedModel(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

    # Split data into train, test and validation sets.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

    # Get base models.
    unfitModels = getUnfitModels()

    # Fit base and stacked models.
    dfPredictions, models = fitBaseModels(X_train, y_train, X_val, unfitModels)
    stackedModel = fitStackedModel(dfPredictions, y_val)

    # Print dataframe used for input to fit the stacked model.
    dfPredictionsCopy = dfPredictions.copy()
    dfPredictionsCopy.columns = [model.__class__.__name__ for model in models]
    print("\n** Dataframe used for stacked model **")
    print(dfPredictionsCopy.head())

    # Evaluate base models with validation data.
    print("\n** Evaluate Base Models **")
    dfValidationPredictions = pd.DataFrame()
    for i in range(0, len(models)):
        predictions = models[i].predict(X_test)
        colName = str(i)
        dfValidationPredictions[colName] = predictions
        evaluateModel(y_test, predictions, models[i])

    # Evaluate stacked model with validation data.
    stackedPredictions = stackedModel.predict(dfValidationPredictions)
    print("\n** Evaluate Stacked Model **")
    evaluateModel(y_test, stackedPredictions, stackedModel)


def ex9():
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Prep data.
    PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
    CSV_DATA = "winequality.csv"
    dataset = pd.read_csv(PATH + CSV_DATA)
    print(dataset.head())
    X = dataset[
        [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        ]
    ].values
    y = dataset["quality"]

    def getUnfitModels():
        models = list()
        models.append(ElasticNet())
        models.append(SVR(gamma="scale"))
        models.append(DecisionTreeRegressor())
        models.append(AdaBoostRegressor())
        models.append(RandomForestRegressor(n_estimators=200))
        models.append(ExtraTreesRegressor(n_estimators=200))
        return models

    def evaluateModel(y_test, predictions, model):
        mse = mean_squared_error(y_test, predictions)
        rmse = round(np.sqrt(mse), 3)
        print(" RMSE:" + str(rmse) + " " + model.__class__.__name__)

    def fitBaseModels(X_train, y_train, X_test, models):
        dfPredictions = pd.DataFrame()

        # Fit base model and store its predictions in dataframe.
        for i in range(0, len(models)):
            models[i].fit(X_train, y_train)
            predictions = models[i].predict(X_test)
            colName = str(i)
            # Add base model predictions to column of data frame.
            dfPredictions[colName] = predictions
        return dfPredictions, models

    def fitStackedModel(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

    # Split data into train, test and validation sets.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

    # Get base models.
    unfitModels = getUnfitModels()

    # Fit base and stacked models.
    dfPredictions, models = fitBaseModels(X_train, y_train, X_val, unfitModels)
    stackedModel = fitStackedModel(dfPredictions, y_val)

    # Evaluate base models with validation data.
    print("\n** Evaluate Base Models **")
    dfValidationPredictions = pd.DataFrame()
    for i in range(0, len(models)):
        predictions = models[i].predict(X_test)
        colName = str(i)
        dfValidationPredictions[colName] = predictions
        evaluateModel(y_test, predictions, models[i])

    # Evaluate stacked model with validation data.
    stackedPredictions = stackedModel.predict(dfValidationPredictions)
    print("\n** Evaluate Stacked Model **")
    evaluateModel(y_test, stackedPredictions, stackedModel)


def ex10():
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Prepare the data.
    PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
    CSV_DATA = "Social_Network_Ads.csv"

    df = pd.read_csv(PATH + CSV_DATA)
    df = pd.get_dummies(df, columns=["Gender"])
    del df["User ID"]

    X = df.copy()
    del X["Purchased"]
    y = df["Purchased"]

    def getUnfitModels():
        models = list()
        models.append(LogisticRegression())
        models.append(DecisionTreeClassifier())
        models.append(AdaBoostClassifier())
        models.append(RandomForestClassifier(n_estimators=10))
        return models

    def evaluateModel(y_test, predictions, model):
        print("\n*** " + model.__class__.__name__)
        report = classification_report(y_test, predictions)
        print(report)

    def fitBaseModels(X_train, y_train, X_test, models):
        dfPredictions = pd.DataFrame()

        # Fit base model and store its predictions in dataframe.
        for i in range(0, len(models)):
            models[i].fit(X_train, y_train)
            predictions = models[i].predict(X_test)
            colName = str(i)
            dfPredictions[colName] = predictions
        return dfPredictions, models

    def fitStackedModel(X, y):
        model = LogisticRegression()
        model.fit(X, y)
        return model

    # Split data into train, test and validation sets.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

    # Get base models.
    unfitModels = getUnfitModels()

    # Fit base and stacked models.
    dfPredictions, models = fitBaseModels(X_train, y_train, X_val, unfitModels)
    stackedModel = fitStackedModel(dfPredictions, y_val)

    # Print dataframe used for input to fit the stacked model.
    dfPredictionsCopy = dfPredictions.copy()
    dfPredictionsCopy.columns = [model.__class__.__name__ for model in models]
    print("\n** Dataframe used for stacked model **")
    print(dfPredictionsCopy.head())

    # Evaluate base models with validation data.
    print("\n** Evaluate Base Models **")
    dfValidationPredictions = pd.DataFrame()
    for i in range(0, len(models)):
        predictions = models[i].predict(X_test)
        colName = str(i)
        dfValidationPredictions[colName] = predictions
        evaluateModel(y_test, predictions, models[i])

    # Evaluate stacked model with validation data.
    stackedPredictions = stackedModel.predict(dfValidationPredictions)
    print("\n** Evaluate Stacked Model **")
    evaluateModel(y_test, stackedPredictions, stackedModel)


def ex11():
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Prep data.
    PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
    CSV_DATA = "iris_v2.csv"
    df = pd.read_csv(PATH + CSV_DATA)
    dict_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    df["target"] = df["iris_type"].map(dict_map)
    print(df.head())

    y = df["target"]
    X = df.copy()
    del X["target"]
    del X["iris_type"]

    def getUnfitModels():
        models = list()
        models.append(ElasticNet())
        models.append(SVR(gamma="scale"))
        models.append(DecisionTreeRegressor())
        models.append(AdaBoostRegressor())
        models.append(RandomForestRegressor(n_estimators=200))
        models.append(ExtraTreesRegressor(n_estimators=200))
        return models

    def evaluateModel(y_test, predictions, model):
        mse = mean_squared_error(y_test, predictions)
        rmse = round(np.sqrt(mse), 3)
        print(" RMSE:" + str(rmse) + " " + model.__class__.__name__)

    def fitBaseModels(X_train, y_train, X_test, models):
        dfPredictions = pd.DataFrame()

        # Fit base model and store its predictions in dataframe.
        for i in range(0, len(models)):
            models[i].fit(X_train, y_train)
            predictions = models[i].predict(X_test)
            colName = str(i)
            # Add base model predictions to column of data frame.
            dfPredictions[colName] = predictions
        return dfPredictions, models

    def fitStackedModel(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

    # Split data into train, test and validation sets.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

    # Get base models.
    unfitModels = getUnfitModels()

    # Fit base and stacked models.
    dfPredictions, models = fitBaseModels(X_train, y_train, X_val, unfitModels)
    stackedModel = fitStackedModel(dfPredictions, y_val)

    # Evaluate base models with validation data.
    print("\n** Evaluate Base Models **")
    dfValidationPredictions = pd.DataFrame()
    for i in range(0, len(models)):
        predictions = models[i].predict(X_test)
        colName = str(i)
        dfValidationPredictions[colName] = predictions
        evaluateModel(y_test, predictions, models[i])

    # Evaluate stacked model with validation data.
    stackedPredictions = stackedModel.predict(dfValidationPredictions)
    print("\n** Evaluate Stacked Model **")
    evaluateModel(y_test, stackedPredictions, stackedModel)


def main():
    # ex1()
    # ex2()
    # ex3()
    # ex5()
    # ex6()
    # ex8()
    # ex9()
    # ex10()
    ex11()


if __name__ == "__main__":
    main()
