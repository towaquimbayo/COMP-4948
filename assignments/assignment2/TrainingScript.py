import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from sklearn.feature_selection import RFE, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    f1_score,
    recall_score,
)
import warnings

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
warnings.filterwarnings("ignore")


def get_data():
    return pd.read_csv("./train.csv")


def prepare_data(df):
    # Drop 'employee_id' because it's a unique identifier
    # Drop 'region' because it's a categorical variable with 34 unique values
    df = df.drop(["employee_id", "region"], axis=1)

    # Check number of duplicates: 0 found
    # print("Number of Duplicates:", df.duplicated().sum())

    # Check for missing values: 'education' has 2409, 'previous_year_rating' has 4124
    # print("Missing Values:")
    # print(df.isnull().sum())

    # Check 'previous_year_rating' rows with null values to find a pattern
    # Pattern: 'previous_year_rating' is null when 'length_of_service' is 1
    # print("Previous Year Rating Rows with Null Values:")
    # df_previous_year_rating_null = df[df["previous_year_rating"].isnull()]
    # print(df_previous_year_rating_null[["length_of_service", "previous_year_rating"]])

    # Check 'education' rows with null values to find a pattern
    # Pattern: No pattern found
    # print("Education Rows with Null Values:")
    # print(df[df["education"].isnull()])

    # Impute missing values in 'previous_year_rating' and 'education'
    df = treat_missing_values(df)

    # Check outliers in numerical features.
    # Note: Other features may have outliers but the values are limited to the range of the feature.
    # Found: only 'length_of_service' has valid outliers
    # print("Outliers:")
    # for feature in df.select_dtypes("number").columns:
    #     plot_boxplot(df, feature)

    # Treat outliers
    df = treat_outliers(df)

    # Check object data types
    # Found: 'department', 'education', 'gender', 'recruitment_channel'
    # print("Object Data Types:")
    # print(df.select_dtypes("object").head())

    # Check unique values of object data types
    # print("Unique Values:")
    # for feature in df.select_dtypes("object").columns:
    #     print(str(df[feature].value_counts()) + "\n")

    # Label encode 'department'
    df["department"] = df["department"].astype("category").cat.codes

    # Label encode 'education'
    df["education"] = df["education"].astype("category").cat.codes

    # Label encode 'gender'
    df["gender"] = df["gender"].astype("category").cat.codes

    # Label encode 'recruitment_channel'
    df["recruitment_channel"] = df["recruitment_channel"].astype("category").cat.codes

    # Check data after preprocessing
    print("Preprocessed Data:")
    print(df.head())
    return df


# Impute missing values
def treat_missing_values(df):
    # Fill 'previous_year_rating' with 0
    df["previous_year_rating"] = df["previous_year_rating"].fillna(0)

    # Fill 'education' with Mode values because it's a categorical ordinal variable
    df["education"] = df["education"].fillna(df["education"].mode()[0])

    # Check missing values after treatment
    # print("Missing Values After Treatment:")
    # print(df.isnull().sum())
    return df


def treat_outliers(df):
    # Treat 'length_of_service' outliers by capping values greater than 13 (Q3)
    df["length_of_service"] = np.where(
        df["length_of_service"] > 13, 13, df["length_of_service"]
    )
    return df


def plot_histograms(df, feature_name, target_name):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 10))
    sns.barplot(x=target_name, y=feature_name, data=df, ci=None)
    plt.title("Barplot of " + feature_name.title() + " vs " + target_name.title())
    plt.xlabel(target_name.title())
    plt.ylabel(feature_name.title())
    plt.tight_layout()
    plt.show()


def plot_boxplot(df, feature_name):
    sns.boxplot(y=feature_name, data=df)
    plt.title("Boxplot of " + feature_name.title())
    plt.xlabel(feature_name.title())
    plt.tight_layout()
    plt.show()


def plot_boxplot_with_target(df, feature_name, target_name):
    sns.boxplot(x=target_name, y=feature_name, data=df)
    plt.title("Boxplot of " + feature_name.title() + " vs " + target_name.title())
    plt.xlabel(target_name.title())
    plt.ylabel(feature_name.title())
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap(df, target_name):
    corr = df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        corr[[target_name]].sort_values(by=target_name, ascending=False),
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        annot=True,
        cmap="YlGnBu",
    )
    plt.title("Correlation Heatmap with " + target_name.title())
    plt.tight_layout()
    plt.show()

    # Correlation matrix
    corr_matrix = df.corr()
    print("Correlation Matrix:")
    print(corr_matrix[target_name].sort_values(ascending=False))


def recursive_feature_elimination(df, features, target, num_features):
    # Split the data into features and target
    x = df[features].drop(target, axis=1)
    x = sm.add_constant(x)
    y = df[target]

    # Initialize the estimator and the RFE model
    estimator = LogisticRegression()
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


def forward_feature_selection(df, features, target):
    x = df[features].drop(target, axis=1)
    x = sm.add_constant(x)
    y = df[target]

    # Initialize the estimator and the Forward Feature Selection model
    f_score, p_value = f_classif(x, y)
    selected_features = []
    features_df = pd.DataFrame(columns=["Feature", "F-Score", "P-Value"])

    print("\nForward Feature Selection:")
    for i, feature in enumerate(x.columns):
        if p_value[i] < 0.05:
            selected_features.append(feature)

        features_df = features_df._append(
            {
                "Feature": feature,
                "F-Score": f_score[i],
                "P-Value": p_value[i],
            },
            ignore_index=True,
        )
    print("Selected Features:", ", ".join(selected_features))
    features_df = features_df.sort_values(by=["F-Score"], ascending=False)
    print(features_df)


def random_forest_regressor_feature_selection(df, features, target):
    # Split the data into features and target
    x = df[features].drop(target, axis=1)
    features_list = list(x.columns)
    x = np.array(x)
    y = np.array(df[target])

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Scale the features
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)

    # Fit the model and print the feature importance's
    model = RandomForestClassifier(n_estimators=1000)
    model.fit(x_train, y_train)
    print("\nRandom Forest Regressor Feature Importance:")

    # Use the forest's predict method on the test data
    predictions = model.predict(x_test)

    # Calculate and display accuracy (returns nan due to division by zero)
    accuracy = accuracy_score(y_test, predictions) * 100
    print("Accuracy:", accuracy, "%.")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Get numerical feature importance's
    importances = list(model.feature_importances_)

    # Show feature importance's
    df_importance = pd.DataFrame({"Importance": importances, "Feature": features_list})
    df_importance = df_importance.sort_values(by=["Importance"], ascending=False)
    print(df_importance)


# Artificial Neural Network
def model_ann(df):
    df = df[
        [
            "department",
            # "education",
            # "gender",
            # "recruitment_channel",
            # "no_of_trainings",
            # "age",
            "previous_year_rating",
            # "length_of_service",
            "awards_won?",
            "avg_training_score",
            "is_promoted",
        ]
    ]
    # Split the data into features and target
    dataset = df.values
    x = dataset[:, 0 : df.shape[1] - 1]
    y = dataset[:, df.shape[1] - 1]
    ROW_DIM = 0
    COL_DIM = 1

    x_array_reshaped = x.reshape(x.shape[ROW_DIM], x.shape[COL_DIM])
    y_array_reshaped = y.reshape(y.shape[ROW_DIM], 1)

    # Split the data into training and testing sets
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_array_reshaped, y_array_reshaped, test_size=0.3
    )
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5)

    # Scale the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)

    # Save the scaler
    with open("model/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    def create_model(
        optimizer,
        learning_rate=0.001,
        initializer="normal",
        neurons=12,
        additional_layers=0,
        activation_function="relu",
    ):
        # Build an Artificial Neural Network model of sequential layers
        model = Sequential()

        # Add first hidden layer (input layer)
        model.add(
            Dense(
                neurons,
                input_dim=x_train.shape[1],
                activation=activation_function,
                kernel_initializer=initializer,
            )
        )

        # Add additional hidden layers
        for i in range(additional_layers):
            model.add(
                Dense(
                    neurons,
                    activation=activation_function,
                    kernel_initializer=initializer,
                )
            )

        # Add output layer
        model.add(Dense(1, kernel_initializer=initializer, activation="sigmoid"))

        # Compile the model
        model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def evaluate_model(model, x_test, y_test):
        # Evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("\n*** Evaluate Artificial Neural Network Model ***")
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

        # Make predictions for the test data
        predictions = model.predict(x_test)
        print("Actual:")
        print(y_test)
        print("Predictions:")
        print(predictions)

        # Convert predictions to binary values and print classification report
        predictions = [1 if x > 0.5 else 0 for x in predictions]
        print("Classification Report:")
        print(classification_report(y_test, predictions))

        precision = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        return loss, accuracy, precision, recall, f1, predictions

    # Grid search for best model parameters
    network_stats = []

    # Model parameters to grid search
    batch_sizes = [10, 50, 100]
    epochs_list = [50, 100, 200]
    optimizers = [Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam]
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    initializers = [
        "uniform",
        "lecun_uniform",
        "normal",
        "zero",
        "glorot_normal",
        "glorot_uniform",
        "he_normal",
        "he_uniform",
    ]
    neuron_list = [5, 25, 50, 100, 150, 200]
    additional_layers = [0, 1, 2, 3, 4, 5]
    activation_functions = [
        "softmax",
        "softplus",
        "softsign",
        "relu",
        "tanh",
        "sigmoid",
    ]

    # Search for best batch size and epochs
    # for batch_size in batch_sizes:
    #     for epochs in epochs_list:
    #         model = create_model(Adam, 0.001, "normal", 12)
    #         history = model.fit(
    #             x_train,
    #             y_train,
    #             epochs=epochs,
    #             batch_size=batch_size,
    #             verbose=1,
    #             validation_data=(x_val, y_val),
    #         )
    #
    #         loss, accuracy, precision, recall, f1, prediction = evaluate_model(
    #             model, x_test, y_test
    #         )
    #         network_stats.append(
    #             {
    #                 "batch_size": batch_size,
    #                 "epochs": epochs,
    #                 "loss": loss,
    #                 "accuracy": accuracy,
    #                 "precision": precision,
    #                 "recall": recall,
    #                 "f1": f1,
    #             }
    #         )

    BEST_EPOCHS = 50
    BEST_BATCH_SIZE = 100

    # Search for best optimizer
    # for optimizer in optimizers:
    #     model = create_model(optimizer, 0.001, "normal", 12)
    #     history = model.fit(
    #         x_train,
    #         y_train,
    #         epochs=BEST_EPOCHS,
    #         batch_size=BEST_BATCH_SIZE,
    #         verbose=1,
    #         validation_data=(x_val, y_val),
    #     )
    #     loss, accuracy, precision, recall, f1, prediction = evaluate_model(model, x_test, y_test)
    #     network_stats.append(
    #         {
    #             "batch_size": BEST_BATCH_SIZE,
    #             "epochs": BEST_EPOCHS,
    #             "optimizer": optimizer.__name__,
    #             "loss": loss,
    #             "accuracy": accuracy,
    #             "precision": precision,
    #             "recall": recall,
    #             "f1": f1,
    #         }
    #     )

    BEST_OPTIMIZER = RMSprop

    # Search for best learning rate
    # for learning_rate in learning_rates:
    #     model = create_model(BEST_OPTIMIZER, learning_rate, "normal", 12)
    #     history = model.fit(
    #         x_train,
    #         y_train,
    #         epochs=BEST_EPOCHS,
    #         batch_size=BEST_BATCH_SIZE,
    #         verbose=1,
    #         validation_data=(x_val, y_val),
    #     )
    #     loss, accuracy, precision, recall, f1, prediction = evaluate_model(model, x_test, y_test)
    #     network_stats.append(
    #         {
    #             "batch_size": BEST_BATCH_SIZE,
    #             "epochs": BEST_EPOCHS,
    #             "optimizer": BEST_OPTIMIZER.__name__,
    #             "learning_rate": learning_rate,
    #             "loss": loss,
    #             "accuracy": accuracy,
    #             "precision": precision,
    #             "recall": recall,
    #             "f1": f1,
    #         }
    #     )

    BEST_LEARNING_RATE = 0.01

    # Search for best kernel initializer
    # for initializer in initializers:
    #     model = create_model(BEST_OPTIMIZER, BEST_LEARNING_RATE, initializer, 12)
    #     history = model.fit(
    #         x_train,
    #         y_train,
    #         epochs=BEST_EPOCHS,
    #         batch_size=BEST_BATCH_SIZE,
    #         verbose=1,
    #         validation_data=(x_val, y_val),
    #     )
    #     loss, accuracy, precision, recall, f1, prediction = evaluate_model(model, x_test, y_test)
    #     network_stats.append(
    #         {
    #             "batch_size": BEST_BATCH_SIZE,
    #             "epochs": BEST_EPOCHS,
    #             "optimizer": BEST_OPTIMIZER.__name__,
    #             "learning_rate": BEST_LEARNING_RATE,
    #             "initializer": initializer,
    #             "loss": loss,
    #             "accuracy": accuracy,
    #             "precision": precision,
    #             "recall": recall,
    #             "f1": f1,
    #         }
    #     )

    BEST_INITIALIZER = "he_normal"

    # Search for best number of neurons
    # for num_neurons in neuron_list:
    #     model = create_model(
    #         BEST_OPTIMIZER, BEST_LEARNING_RATE, BEST_INITIALIZER, num_neurons
    #     )
    #     history = model.fit(
    #         x_train,
    #         y_train,
    #         epochs=BEST_EPOCHS,
    #         batch_size=BEST_BATCH_SIZE,
    #         verbose=1,
    #         validation_data=(x_val, y_val),
    #     )
    #     loss, accuracy, precision, recall, f1, prediction = evaluate_model(model, x_test, y_test)
    #     network_stats.append(
    #         {
    #             "batch_size": BEST_BATCH_SIZE,
    #             "epochs": BEST_EPOCHS,
    #             "optimizer": BEST_OPTIMIZER.__name__,
    #             "learning_rate": BEST_LEARNING_RATE,
    #             "initializer": BEST_INITIALIZER,
    #             "neurons": num_neurons,
    #             "loss": loss,
    #             "accuracy": accuracy,
    #             "precision": precision,
    #             "recall": recall,
    #             "f1": f1,
    #         }
    #     )

    BEST_NEURONS = 150

    # Search for best number of additional layers
    # for num_layers in additional_layers:
    #     model = create_model(
    #         BEST_OPTIMIZER,
    #         BEST_LEARNING_RATE,
    #         BEST_INITIALIZER,
    #         BEST_NEURONS,
    #         num_layers,
    #     )
    #     history = model.fit(
    #         x_train,
    #         y_train,
    #         epochs=BEST_EPOCHS,
    #         batch_size=BEST_BATCH_SIZE,
    #         verbose=1,
    #         validation_data=(x_val, y_val),
    #     )
    #     loss, accuracy, precision, recall, f1, prediction = evaluate_model(model, x_test, y_test)
    #     network_stats.append(
    #         {
    #             "batch_size": BEST_BATCH_SIZE,
    #             "epochs": BEST_EPOCHS,
    #             "optimizer": BEST_OPTIMIZER.__name__,
    #             "learning_rate": BEST_LEARNING_RATE,
    #             "initializer": BEST_INITIALIZER,
    #             "neurons": BEST_NEURONS,
    #             "additional_layers": num_layers,
    #             "loss": loss,
    #             "accuracy": accuracy,
    #             "precision": precision,
    #             "recall": recall,
    #             "f1": f1,
    #         }
    #     )

    BEST_ADDITIONAL_LAYERS = 3

    # Search for best activation function
    # for activation_function in activation_functions:
    #     model = create_model(
    #         BEST_OPTIMIZER,
    #         BEST_LEARNING_RATE,
    #         BEST_INITIALIZER,
    #         BEST_NEURONS,
    #         BEST_ADDITIONAL_LAYERS,
    #         activation_function,
    #     )
    #     history = model.fit(
    #         x_train,
    #         y_train,
    #         epochs=BEST_EPOCHS,
    #         batch_size=BEST_BATCH_SIZE,
    #         verbose=1,
    #         validation_data=(x_val, y_val),
    #     )
    #     loss, accuracy, precision, recall, f1, prediction = evaluate_model(
    #         model, x_test, y_test
    #     )
    #     network_stats.append(
    #         {
    #             "batch_size": BEST_BATCH_SIZE,
    #             "epochs": BEST_EPOCHS,
    #             "optimizer": BEST_OPTIMIZER.__name__,
    #             "learning_rate": BEST_LEARNING_RATE,
    #             "initializer": BEST_INITIALIZER,
    #             "neurons": BEST_NEURONS,
    #             "additional_layers": BEST_ADDITIONAL_LAYERS,
    #             "activation_function": activation_function,
    #             "loss": loss,
    #             "accuracy": accuracy,
    #             "precision": precision,
    #             "recall": recall,
    #             "f1": f1,
    #         }
    #     )

    BEST_ACTIVATION_FUNCTION = "softsign"

    # Print network stats and parameters of the best model
    # df_network_stats = pd.DataFrame(network_stats)
    # df_network_stats = df_network_stats.sort_values(by=["f1"], ascending=False)
    # print(df_network_stats)

    def cross_fold_validation(x, y):
        k_fold = KFold(n_splits=10, shuffle=True)
        accuracy_list = list()
        precision_list = list()
        recall_list = list()
        f1_list = list()

        print("\nCross Fold Validation:")
        for train_index, val_index in k_fold.split(x):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Scale the features
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)

            # Build the ANN model
            model = create_model(
                BEST_OPTIMIZER,
                BEST_LEARNING_RATE,
                BEST_INITIALIZER,
                BEST_NEURONS,
                BEST_ADDITIONAL_LAYERS,
                BEST_ACTIVATION_FUNCTION,
            )

            # Fit the model
            history = model.fit(
                x_train,
                y_train,
                epochs=BEST_EPOCHS,
                batch_size=BEST_BATCH_SIZE,
                verbose=1,
                validation_data=(x_val, y_val),
            )

            # Evaluate the model
            loss, accuracy, precision, recall, f1, predictions = evaluate_model(
                model, x_val, y_val
            )
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        # Print average metrics of the model across all folds
        print("\nAverage Accuracy:", np.mean(accuracy_list))
        print("Standard Deviation of Accuracy:", np.std(accuracy_list))
        print("Average Precision:", np.mean(precision_list))
        print("Standard Deviation of Precision:", np.std(precision_list))
        print("Average Recall:", np.mean(recall_list))
        print("Standard Deviation of Recall:", np.std(recall_list))
        print("Average F1 Score:", np.mean(f1_list))
        print("Standard Deviation of F1 Score:", np.std(f1_list))

    # Cross fold validation
    cross_fold_validation(x, y)

    # Build the stand-alone ANN model
    print("\n*** Stand-Alone Model Evaluation ***")
    model = create_model(
        BEST_OPTIMIZER,
        BEST_LEARNING_RATE,
        BEST_INITIALIZER,
        BEST_NEURONS,
        BEST_ADDITIONAL_LAYERS,
        BEST_ACTIVATION_FUNCTION,
    )

    # Create early stopping and model checkpoint callbacks
    es = EarlyStopping(
        monitor="val_loss", min_delta=0.000001, mode="min", verbose=1, patience=200
    )
    mc = ModelCheckpoint(
        "model/best_model.keras",
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )

    # Fit the model
    history = model.fit(
        x_train,
        y_train,
        epochs=BEST_EPOCHS,
        batch_size=BEST_BATCH_SIZE,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=[es, mc],
    )

    # Evaluate the model
    evaluate_model(model, x_test, y_test)

    def show_loss(history):
        # Get training and test loss histories
        training_loss = history.history["loss"]
        validation_loss = history.history["val_loss"]

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)
        plt.subplot(1, 2, 1)

        # Visualize loss history for training data
        plt.plot(epoch_count, training_loss, label="Train Loss", color="red")

        # View loss on test data
        plt.plot(
            epoch_count, validation_loss, "r--", label="Validation Loss", color="black"
        )

        plt.xlabel("Epoch")
        plt.legend(loc="best")
        plt.title("Loss History")
        plt.tight_layout()

    def show_accuracy(history):
        # Get training and test loss histories
        training_accuracy = history.history["accuracy"]
        validation_accuracy = history.history["val_accuracy"]

        # Create count of the number of epochs
        epoch_count = range(1, len(training_accuracy) + 1)
        plt.subplot(1, 2, 2)

        # Visualize loss history for training data
        plt.plot(epoch_count, training_accuracy, label="Train Accuracy", color="red")

        # View loss on test data
        plt.plot(
            epoch_count,
            validation_accuracy,
            "r--",
            label="Validation Accuracy",
            color="black",
        )

        plt.xlabel("Epoch")
        plt.legend(loc="best")
        plt.title("Accuracy History")
        plt.tight_layout()

    # Show loss and accuracy history
    plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    show_loss(history)
    show_accuracy(history)
    plt.tight_layout()
    plt.show()


# Stacked Model (Logistic Regression, Decision Tree, AdaBoost, Random Forest)
def model_stacked(df, target):
    # Split the data into features and target
    x = df[
        [
            "department",
            # "education",
            # "gender",
            # "recruitment_channel",
            # "no_of_trainings",
            # "age",
            "previous_year_rating",
            "length_of_service",
            # "awards_won?",
            "avg_training_score",
        ]
    ]
    y = df[target]

    def get_unfit_models():
        models = list()
        models.append(LogisticRegression())
        models.append(DecisionTreeClassifier())
        models.append(AdaBoostClassifier())
        models.append(RandomForestClassifier(n_estimators=200))
        return models

    def evaluate_model(y_test, predictions, model):
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")
        print("Model:", model.__class__.__name__)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Classification Report:")
        print(classification_report(y_test, predictions))

    def fit_base_models(x_train, y_train, x_test, models):
        # Fit the base models and store its predictions in a dataframe
        df_predictions = pd.DataFrame()
        for i in range(len(models)):
            models[i].fit(x_train, y_train)
            predictions = models[i].predict(x_test)
            df_predictions[i] = predictions
        return df_predictions, models

    def fit_stacked_model(x, y):
        model = LogisticRegression()
        model.fit(x, y)
        return model

    def cross_fold_validation(x, y):
        k_fold = KFold(n_splits=10, shuffle=True)
        accuracy_list = list()
        precision_list = list()
        recall_list = list()
        f1_list = list()

        print("\nCross Fold Validation:")
        for train_index, val_index in k_fold.split(x):
            x_train, x_val = x.iloc[train_index], x.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Get the base models
            unfit_models = get_unfit_models()

            # Fit the base models and stacked model
            df_predictions, models = fit_base_models(
                x_train, y_train, x_val, unfit_models
            )
            stacked_model = fit_stacked_model(df_predictions, y_val)

            # Evaluate the base models and validation data
            print("\n*** Evaluate Base Models ***")
            df_validation_predictions = pd.DataFrame()
            for i in range(len(models)):
                predictions = models[i].predict(x_val)
                df_validation_predictions[i] = predictions
                evaluate_model(y_val, predictions, models[i])

            # Evaluate the stacked model and validation data
            print("\n*** Evaluate Stacked Model ***")
            stacked_predictions = stacked_model.predict(df_validation_predictions)
            evaluate_model(y_val, stacked_predictions, stacked_model)

            accuracy_list.append(accuracy_score(y_val, stacked_predictions))
            precision_list.append(
                precision_score(y_val, stacked_predictions, average="weighted")
            )
            recall_list.append(
                recall_score(y_val, stacked_predictions, average="weighted")
            )
            f1_list.append(f1_score(y_val, stacked_predictions, average="weighted"))

        # Print average metrics of the stacked model across all folds
        print("\nAverage Accuracy:", np.mean(accuracy_list))
        print("Standard Deviation of Accuracy:", np.std(accuracy_list))
        print("Average Precision:", np.mean(precision_list))
        print("Standard Deviation of Precision:", np.std(precision_list))
        print("Average Recall:", np.mean(recall_list))
        print("Standard Deviation of Recall:", np.std(recall_list))
        print("Average F1 Score:", np.mean(f1_list))
        print("Standard Deviation of F1 Score:", np.std(f1_list))

    # Cross fold validation
    cross_fold_validation(x, y)

    # Stand-alone model evaluation
    print("\n*** Stand-Alone Model Evaluation ***")

    # Split the data into training, testing and validation sets
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.7)
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5)

    # Get the base models
    unfit_models = get_unfit_models()

    # Fit the base models and stacked model
    df_predictions, models = fit_base_models(x_train, y_train, x_val, unfit_models)
    stacked_model = fit_stacked_model(df_predictions, y_val)

    # Evaluate the base models and validation data
    print("\n*** Evaluate Base Models ***")
    df_validation_predictions = pd.DataFrame()
    for i in range(len(models)):
        predictions = models[i].predict(x_test)
        df_validation_predictions[i] = predictions
        evaluate_model(y_test, predictions, models[i])

    # Evaluate the stacked model and validation data
    print("\n*** Evaluate Stacked Model ***")
    stacked_predictions = stacked_model.predict(df_validation_predictions)
    evaluate_model(y_test, stacked_predictions, stacked_model)


# Logistic Regression
def model_logistic(df, target):
    # Split the data into features and target
    x = df[
        [
            # "department",
            # "education",
            # "gender",
            # "recruitment_channel",
            "no_of_trainings",
            # "age",
            "previous_year_rating",
            # "length_of_service",
            "awards_won?",
            "avg_training_score",
        ]
    ]
    y = df[target]

    def cross_fold_validation(x, y):
        k_fold = KFold(n_splits=10, shuffle=True)
        accuracy_list = list()
        precision_list = list()
        recall_list = list()
        f1_list = list()

        print("\nCross Fold Validation:")
        for train_index, val_index in k_fold.split(x):
            x_train, x_val = x.iloc[train_index], x.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Scale the features
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)

            # Build a Logistic Regression model
            model = LogisticRegression(fit_intercept=True, solver="liblinear")
            model.fit(x_train, y_train)

            # Evaluate the model
            predictions = model.predict(x_val)
            accuracy = accuracy_score(y_val, predictions)
            precision = precision_score(y_val, predictions, average="weighted")
            recall = recall_score(y_val, predictions, average="weighted")
            f1 = f1_score(y_val, predictions, average="weighted")
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            print("Classification Report:")
            print(classification_report(y_val, predictions))

            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        # Print average metrics of the model across all folds
        print("\nAverage Accuracy:", np.mean(accuracy_list))
        print("Standard Deviation of Accuracy:", np.std(accuracy_list))
        print("Average Precision:", np.mean(precision_list))
        print("Standard Deviation of Precision:", np.std(precision_list))
        print("Average Recall:", np.mean(recall_list))
        print("Standard Deviation of Recall:", np.std(recall_list))
        print("Average F1 Score:", np.mean(f1_list))
        print("Standard Deviation of F1 Score:", np.std(f1_list))

    # Cross fold validation
    cross_fold_validation(x, y)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Scale the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Build a Logistic Regression model
    model = LogisticRegression(fit_intercept=True, solver="liblinear")
    model.fit(x_train, y_train)

    # Evaluate the model
    predictions = model.predict(x_test)
    print("\n*** Evaluate Logistic Regression Model ***")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:")
    print(classification_report(y_test, predictions))


def main():
    # Get the data
    df = get_data()

    # Print data information
    print("Data Information:")
    print(df.head())
    print(df.tail())
    print(df.info())
    print(df.shape)
    print(df.describe())

    # Prepare data that imputes missing values and treats outliers
    df = prepare_data(df)

    # Best features
    # features = [
    #     "department",
    #     "previous_year_rating",
    #     "awards_won?",
    #     "avg_training_score",
    #     "is_promoted",
    # ]

    # Plot histograms of features with target
    # for feature in [column for column in features if column != "is_promoted"]:
    #     plot_histograms(df, feature, "is_promoted")

    # Plot boxplot of features with target
    # for feature in [column for column in features if column != "is_promoted"]:
    #     plot_boxplot_with_target(df, feature, "is_promoted")

    # Plot correlation heatmap
    # plot_corr_heatmap(df[features], "is_promoted")

    # Recursive Feature Elimination
    # recursive_feature_elimination(
    #     df,
    #     df.columns,
    #     "is_promoted",
    #     5,
    # )

    # Forward Feature Selection
    # forward_feature_selection(
    #     df,
    #     df.columns,
    #     "is_promoted",
    # )

    # Random Forest Regressor Feature Selection
    # random_forest_regressor_feature_selection(
    #     df,
    #     df.columns,
    #     "is_promoted",
    # )

    # Model: Artificial Neural Network
    model_ann(df)

    # Model: Stacked Model
    model_stacked(df, "is_promoted")

    # Model: Logistic Regression
    model_logistic(df, "is_promoted")


if __name__ == "__main__":
    main()
