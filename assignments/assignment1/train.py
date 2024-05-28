import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import RFE, f_regression
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from sklearn.tree import DecisionTreeRegressor

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)


def get_data():
    path = "/Users/elber/Documents/COMP 4948 - Datasets/assignment1/"
    return pd.read_csv(path + "train.csv")


def plot_histograms(df, feature_name):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 10))
    sns.barplot(x="price_range", y=feature_name, data=df, ci=None)
    plt.title("Barplot of " + feature_name.title() + " vs Price Range")
    plt.xlabel("Price Range")
    plt.ylabel(feature_name.title())
    plt.tight_layout()
    plt.show()


def plot_scatter_matrix(df, feature_name):
    plt.figure(figsize=(10, 10))
    plt.scatter(df[feature_name], df["price_range"])
    plt.title("Scatter Plot of Price Range vs " + feature_name.title())
    plt.xlabel(feature_name.title())
    plt.ylabel("Price Range")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_boxplot(df, feature_name):
    sns.boxplot(x="price_range", y=feature_name, data=df)
    plt.title("Boxplot of " + feature_name.title() + " vs Price Range")
    plt.xlabel("Price Range")
    plt.ylabel(feature_name.title())
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap(df):
    corr = df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        corr[["price_range"]].sort_values(by="price_range", ascending=False),
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        annot=True,
        cmap="YlGnBu",
    )
    plt.title("Correlation Heatmap with Price Range")
    plt.tight_layout()
    plt.show()

    # Correlation matrix
    corr_matrix = df.corr()
    print("Correlation Matrix:")
    print(corr_matrix["price_range"].sort_values(ascending=False))


def plot_kmeans_clustering(df, best_features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[best_features])

    # KMeans Clustering
    k_values = [2, 3, 4]
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        df[f"kmeans_{k}"] = kmeans.fit_predict(scaled_features)

    # Analyze the characteristics of each segment
    for k in k_values:
        print(f"\n*** K-means with {k} clusters ***")
        for cluster in range(k):
            segment_data = df[df[f"kmeans_{k}"] == cluster][
                ["battery_power", "px_height", "px_width", "ram"]
            ]
            print(f"\nSegment {cluster + 1}:")
            print(segment_data.describe())

    # Plot the clusters
    for k in k_values:
        plt.figure(figsize=(10, 10))
        plt.title(f"K-means with {k} clusters")
        plt.scatter(
            # df["ram"],
            # df["battery_power"],
            df["px_height"],
            df["px_width"],
            c=df[f"kmeans_{k}"],
            cmap="viridis",
        )
        plt.xlabel("Px Height")
        plt.ylabel("Px Width")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


def cross_fold_validation(x, y, model_type):
    k_fold = KFold(n_splits=10, shuffle=True)
    rmse_list = []
    accuracy_list = []
    print("\nCross Fold Validation:")
    for train_index, test_index in k_fold.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Build the model and make predictions
        if model_type == "ols":
            model = sm.OLS(y_train, x_train).fit()
            predictions = model.predict(x_test)
            print(model.summary())

            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            print("Root Mean Squared Error:", rmse)
            rmse_list.append(rmse)

            accuracy = explained_variance_score(y_test, predictions)
            print("Accuracy:", accuracy)
            accuracy_list.append(accuracy)
        elif model_type == "bagging_regressor":
            model = BaggingRegressor(
                estimator=LinearRegression(),
                max_samples=0.5,
                max_features=1,
                n_estimators=100,
            )
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)

            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            print("Root Mean Squared Error:", rmse)
            rmse_list.append(rmse)

            accuracy = model.score(x_test, y_test)
            print("Accuracy:", accuracy)
            accuracy_list.append(accuracy)
        elif model_type == "ensemble_regressor":
            # Scale the features
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Define the standalone models
            svr = SVR(kernel="linear")
            lasso = Lasso()
            rf = RandomForestRegressor(n_estimators=5)
            models = [svr, lasso, rf]

            # Fit standalone models
            print()
            for i in range(len(models)):
                models[i].fit(x_train, y_train)
                predictions = models[i].predict(x_test)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                print(
                    "Model Type:",
                    models[i].__class__.__name__,
                    "Root Mean Squared Error:",
                    rmse,
                )

            # Build stack model with prefit models
            stack = StackingCVRegressor(
                regressors=(models[0], models[1], models[2]),
                meta_regressor=LinearRegression(),
            )
            stack.fit(x_train, y_train)
            predictions = stack.predict(x_test)

            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            print("Ensemble Regressor Model - Root Mean Squared Error:", rmse)
            rmse_list.append(rmse)

            accuracy = stack.score(x_test, y_test)
            print("Ensemble Regressor Model - Accuracy:", accuracy)
            accuracy_list.append(accuracy)

    print("\nAverage RMSE:", np.mean(rmse_list))
    print("Standard Deviation RMSE:", np.std(rmse_list))
    print("Average Accuracy:", np.mean(accuracy_list))
    print("Standard Deviation Accuracy:", np.std(accuracy_list))


def recursive_feature_elimination(df, num_features):
    # Split the data into features and target
    x = df[
        [
            "battery_power",
            "blue",
            "clock_speed",
            "dual_sim",
            "fc",
            "four_g",
            "int_memory",
            "m_dep",
            "mobile_wt",
            "n_cores",
            "pc",
            "px_height",
            "px_width",
            "ram",
            "sc_h",
            "sc_w",
            "talk_time",
            "three_g",
            "touch_screen",
            "wifi",
        ]
    ]
    x = sm.add_constant(x)
    y = df["price_range"]

    # Initialize the estimator and the RFE model
    estimator = LinearRegression()
    rfe = RFE(estimator, n_features_to_select=num_features, step=1)
    rfe = rfe.fit(x, y)
    print("\nRecursive Feature Elimination:")

    # Print the features and their rankings
    feature_names = x.columns[rfe.support_].tolist()
    print("Selected Features:", ", ".join(feature_names))

    sorted_features = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), x.columns))
    print("Sorted Features:")
    for i in sorted_features:
        print(i)


def forward_feature_selection(df):
    x = df[
        [
            "battery_power",
            "blue",
            "clock_speed",
            "dual_sim",
            "fc",
            "four_g",
            "int_memory",
            "m_dep",
            "mobile_wt",
            "n_cores",
            "pc",
            "px_height",
            "px_width",
            "ram",
            "sc_h",
            "sc_w",
            "talk_time",
            "three_g",
            "touch_screen",
            "wifi",
        ]
    ]
    x = sm.add_constant(x)
    y = df["price_range"]

    # Initialize the estimator and the Forward Feature Selection model
    ffs = f_regression(x, y)
    features_df = pd.DataFrame()
    print("\nForward Feature Selection:")

    for i in range(0, len(x.columns)):
        features_df = features_df._append(
            {
                "Feature": x.columns[i],
                "F-Score": ffs[0][i],
                "P-Value": ffs[1][i],
            },
            ignore_index=True,
        )
    features_df = features_df.sort_values(by=["F-Score"], ascending=False)
    print(features_df)


def random_forest_regressor_feature_selection(df):
    # Split the data into features and target
    x = df[
        [
            "battery_power",
            "blue",
            "clock_speed",
            "dual_sim",
            "fc",
            "four_g",
            "int_memory",
            "m_dep",
            "mobile_wt",
            "n_cores",
            "pc",
            "px_height",
            "px_width",
            "ram",
            "sc_h",
            "sc_w",
            "talk_time",
            "three_g",
            "touch_screen",
            "wifi",
        ]
    ]
    features_list = list(x.columns)
    x = np.array(x)
    y = np.array(df["price_range"])

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Scale the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Fit the model and print the feature importances
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(x_train, y_train)
    print("\nRandom Forest Regressor Feature Importance:")

    # Use the forest's predict method on the test data
    predictions = model.predict(x_test)

    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    print("Mean Absolute Error:", np.mean(errors), "degrees.")

    # Calculate and display accuracy (returns nan due to division by zero)
    accuracy = model.score(x_test, y_test)
    print("Accuracy:", accuracy, "%.")

    # Print RMSE
    print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))

    # Get numerical feature importances
    importances = list(model.feature_importances_)

    def show_feature_importances(importances, feature_list):
        df_importance = pd.DataFrame()
        for i in range(0, len(importances)):
            df_importance = df_importance._append(
                {"Importance": importances[i], "Feature": feature_list[i]},
                ignore_index=True,
            )
        df_importance = df_importance.sort_values(by=["Importance"], ascending=False)
        print(df_importance)

    # Show feature importances
    show_feature_importances(importances, features_list)


def model_bagging_regressor(df):
    # Split the data into features and target
    x = df[
        [
            # "battery_power",
            # "blue",
            # "clock_speed",
            # "dual_sim",
            # "fc",
            # "four_g",
            # "int_memory",
            # "m_dep",
            # "mobile_wt",
            # "n_cores",
            # "pc",
            # "px_height",
            # "px_width",
            "ram",
            # "sc_h",
            # "sc_w",
            # "talk_time",
            # "three_g",
            # "touch_screen",
            # "wifi",
        ]
    ]
    x = sm.add_constant(x)
    y = df["price_range"]

    # Cross fold validation
    cross_fold_validation(x, y, "bagging_regressor")

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Evaluate the model
    def evaluate_model(model, x_test, y_test, title):
        print("\n*****", title)
        predictions = model.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print("Root Mean Squared Error:", rmse)
        if title == "Bagging Ensemble":
            print("Accuracy:", model.score(x_test, y_test))
            print("Estimator:", model.estimator)
            print("Estimator Features:", model.estimators_features_)
            print("Estimator Parameters:", model.get_params())

    # Fit the model, print the summary and calculate the RMSE
    model = BaggingRegressor(
        estimator=LinearRegression(), max_samples=0.5, max_features=1, n_estimators=1000
    )
    model.fit(x_train, y_train)
    evaluate_model(model, x_test, y_test, "Bagging Ensemble")

    # Build standalone linear regression model
    model = LinearRegression()
    model.fit(x_train, y_train)
    evaluate_model(model, x_test, y_test, "Linear Regression")


def model_ensemble_regressor(df):
    # Split the data into features and target
    x = df[
        [
            "battery_power",
            # "blue",
            # "clock_speed",
            # "dual_sim",
            # "fc",
            # "four_g",
            # "int_memory",
            # "m_dep",
            # "mobile_wt",
            # "n_cores",
            # "pc",
            "px_height",
            "px_width",
            "ram",
            # "sc_h",
            # "sc_w",
            # "talk_time",
            # "three_g",
            # "touch_screen",
            # "wifi",
        ]
    ]
    x = sm.add_constant(x)
    y = df["price_range"]

    # Cross fold validation
    cross_fold_validation(x, y, "ensemble_regressor")

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Scale the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Define the standalone models
    svr = SVR(kernel="linear")
    lasso = Lasso()
    rf = RandomForestRegressor(n_estimators=5)
    models = [svr, lasso, rf]

    # Fit standalone models
    print()
    for model in models:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(
            "Model Type:",
            model.__class__.__name__,
            " - Root Mean Squared Error:",
            rmse,
        )

    # Build stack model with prefit models
    stack = StackingCVRegressor(
        regressors=(models[0], models[1], models[2]), meta_regressor=LinearRegression()
    )
    stack.fit(x_train, y_train)
    predictions = stack.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("Ensemble Regressor Model - Root Mean Squared Error:", rmse)
    accuracy = stack.score(x_test, y_test)
    print("Ensemble Regressor Model - Accuracy:", accuracy)


def model_stacked_regressor(df):
    # Split the data into features and target
    x = df[
        [
            "battery_power",
            # "blue",
            # "clock_speed",
            # "dual_sim",
            # "fc",
            # "four_g",
            # "int_memory",
            # "m_dep",
            # "mobile_wt",
            # "n_cores",
            # "pc",
            "px_height",
            "px_width",
            "ram",
            # "sc_h",
            # "sc_w",
            # "talk_time",
            # "three_g",
            # "touch_screen",
            # "wifi",
        ]
    ]
    x = sm.add_constant(x)
    y = df["price_range"]

    def get_unfit_models():
        models = list()
        models.append(ElasticNet())
        models.append(SVR(gamma="scale"))
        models.append(AdaBoostRegressor())
        models.append(DecisionTreeRegressor())
        models.append(RandomForestRegressor(n_estimators=200))
        models.append(BaggingRegressor(n_estimators=200))
        models.append(ExtraTreesRegressor(n_estimators=200))
        return models

    def evaluate_model(x_test, y_test, predictions, model):
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print("Model:", model.__class__.__name__)
        print("Root Mean Squared Error:", rmse)
        print("Accuracy:", model.score(x_test, y_test))

    def fit_base_models(x_train, y_train, x_test, models):
        # Fit the base models and store its predictions in a dataframe
        df_predictions = pd.DataFrame()
        for i in range(len(models)):
            models[i].fit(x_train, y_train)
            predictions = models[i].predict(x_test)
            df_predictions[i] = predictions
        return df_predictions, models

    def fit_stacked_model(x, y):
        model = LinearRegression()
        model.fit(x, y)
        return model

    # Cross fold validation
    k_fold = KFold(n_splits=10, shuffle=True)
    rmse_list = []
    accuracy_list = []
    print("\nCross Fold Validation:")
    for train_index, val_index in k_fold.split(x):
        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Get the base models
        unfit_models = get_unfit_models()

        # Fit the base models and stacked model
        df_predictions, models = fit_base_models(x_train, y_train, x_val, unfit_models)
        stacked_model = fit_stacked_model(df_predictions, y_val)

        # Evaluate the base models and validation data
        print("\n** Evaluate Base Models ***")
        df_validation_predictions = pd.DataFrame()
        for i in range(len(models)):
            predictions = models[i].predict(x_val)
            df_validation_predictions[i] = predictions
            evaluate_model(x_val, y_val, predictions, models[i])

        # Evaluate the stacked model and validation data
        print("\n** Evaluate Stacked Model ***")
        stacked_predictions = stacked_model.predict(df_validation_predictions)
        evaluate_model(
            df_validation_predictions, y_val, stacked_predictions, stacked_model
        )

        rmse_list.append(np.sqrt(mean_squared_error(y_val, stacked_predictions)))
        accuracy_list.append(stacked_model.score(df_validation_predictions, y_val))

    print("\nAverage RMSE:", np.mean(rmse_list))
    print("Standard Deviation RMSE:", np.std(rmse_list))
    print("Average Accuracy:", np.mean(accuracy_list))
    print("Standard Deviation Accuracy:", np.std(accuracy_list))

    # Standalone Model Evaluation
    print("\nStandalone Model Evaluation:")
    # Split the data into training, testing and validation sets
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.7)
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5)

    # Get the base models
    unfit_models = get_unfit_models()

    # Fit the base models and stacked model
    df_predictions, models = fit_base_models(x_train, y_train, x_val, unfit_models)
    stacked_model = fit_stacked_model(df_predictions, y_val)

    # Evaluate the base models and validation data
    print("\n** Evaluate Base Models ***")
    df_validation_predictions = pd.DataFrame()
    for i in range(len(models)):
        predictions = models[i].predict(x_test)
        df_validation_predictions[i] = predictions
        evaluate_model(x_test, y_test, predictions, models[i])

    # Evaluate the stacked model and validation data
    print("\n** Evaluate Stacked Model ***")
    stacked_predictions = stacked_model.predict(df_validation_predictions)
    evaluate_model(
        df_validation_predictions, y_test, stacked_predictions, stacked_model
    )


def model_ols(df):
    # Split the data into features and target
    x = df[
        [
            "battery_power",
            # "blue",
            # "clock_speed",
            # "dual_sim",
            # "fc",
            # "four_g",
            # "int_memory",
            # "m_dep",
            # "mobile_wt",
            # "n_cores",
            # "pc",
            # "px_height",
            "px_width",
            "ram",
            # "sc_h",
            # "sc_w",
            # "talk_time",
            # "three_g",
            # "touch_screen",
            # "wifi",
        ]
    ]
    x = sm.add_constant(x)
    y = df["price_range"]

    # Cross fold validation
    cross_fold_validation(x, y, "ols")

    # Reset y
    y = df["price_range"]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Scale the features
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)
    # y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))

    # Add constant to the features
    # x_train = sm.add_constant(x_train)
    # x_test = sm.add_constant(x_test)

    # Fit the model, print the summary and calculate the RMSE
    model = sm.OLS(y_train, x_train).fit()
    print(model.summary())
    predictions = model.predict(x_test)
    # predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))
    print("Accuracy:", explained_variance_score(y_test, predictions))


def main():
    df = get_data()
    best_features = ["battery_power", "px_height", "px_width", "ram"]

    print(df.shape)
    print(df.head())
    print(df.tail())
    print(df.info())

    # Plot histogram for each column.
    # [plot_histograms(df, column) for column in best_features]

    # Scatter plot for each column.
    # [plot_scatter_matrix(df, column) for column in best_features]

    # Boxplot for each column.
    # [plot_boxplot(df, column) for column in best_features]

    # Plot correlation heatmap
    # plot_corr_heatmap(df[best_features + ["price_range"]])

    # K-means clustering
    # plot_kmeans_clustering(df, best_features)

    # Feature Selection #1: Recursive Feature Elimination
    # recursive_feature_elimination(df, 3)

    # Feature Selection #2: Forward Feature Selection
    # forward_feature_selection(df)

    # Feature Selection #3: RandomForest Regressor feature importance
    # random_forest_regressor_feature_selection(df)

    # Build Bagging Regressor Model
    model_bagging_regressor(df)

    # Build Ensemble mlxtend.regressor Model
    model_ensemble_regressor(df)

    # Build Stacked Regressor Model
    model_stacked_regressor(df)

    # Build OLS Linear Regression Model
    model_ols(df)


if __name__ == "__main__":
    main()
