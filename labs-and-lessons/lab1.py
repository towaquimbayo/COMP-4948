def ex1():
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm

    PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
    CSV_DATA = "winequality.csv"

    dataset = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,  # Don't include header row as part of data.
        encoding="ISO-8859-1",
        sep=",",
        names=(
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
            "quality",
        ),
    )
    # Show all columns.
    pd.set_option("display.max_columns", None)

    # Increase number of columns that display on one line.
    pd.set_option("display.width", 1000)
    print(dataset.head())
    print(dataset.describe())
    X = dataset[
        [
            # "fixed acidity",  # insignificant
            "volatile acidity",
            # "citric acid",  # insignificant
            # "residual sugar",  # insignificant
            "chlorides",
            # "free sulfur dioxide",  # insignificant
            "total sulfur dioxide",
            # "density",  # insignificant
            # "pH",  # insignificant
            "sulphates",
            "alcohol",
        ]
    ].values

    # Adding an intercept *** This is required ***. Don't forget this step.
    # The intercept centers the error residuals around zero
    # which helps to avoid over-fitting.
    X = sm.add_constant(X)
    y = dataset["quality"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model
    print(model.summary())
    print(
        "Root Mean Squared Error:",
        np.sqrt(metrics.mean_squared_error(y_test, predictions)),
    )


def ex7():
    weights = [0.5, 2.3, 2.9]
    heights = [1.4, 1.9, 3.2]

    def getSlopeOfLossFunction(weights, heights, intercept):
        sum = 0
        BETA = 0.64
        for i in range(0, len(weights)):
            sum += -2 * (heights[i] - intercept - BETA * weights[i])

        print("Intercept: " + str(intercept) + " Res: " + str(round(sum, 2)))

    intercept = 0.95
    getSlopeOfLossFunction(weights, heights, intercept)


def main():
    # ex1()
    ex7()


if __name__ == "__main__":
    main()
