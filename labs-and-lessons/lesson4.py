def ex3():
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam

    # Read the data.
    PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
    CSV_DATA = "housing.data"
    df = pd.read_csv(PATH + CSV_DATA, header=None)

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head())
    print(df.tail())
    print(df.describe())

    # Convert DataFrame columns to vertical columns so they can be used by the NN.
    dataset = df.values
    X = dataset[:, 0:13]  # Columns 0 to 12
    y = dataset[:, 13]  # Columns 13
    ROW_DIM = 0
    COL_DIM = 1

    x_arrayReshaped = X.reshape(X.shape[ROW_DIM], X.shape[COL_DIM])
    y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)

    # Split the data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        x_arrayReshaped, y_arrayReshaped, test_size=0.3, random_state=0
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=0
    )

    # Define the model.
    def create_model():
        model = Sequential()
        model.add(
            Dense(13, input_dim=13, kernel_initializer="normal", activation="relu")
        )
        model.add(Dense(1, kernel_initializer="normal"))
        optimizer = Adam(lr=0.005)
        model.compile(loss="mean_squared_error", optimizer=optimizer)
        return model

    model = create_model()

    # Build the model.
    model = create_model()
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=10,
        verbose=1,
        validation_data=(X_val, y_val),
    )

    # Evaluate the model.
    predictions = model.predict(X_test)
    mse = metrics.mean_squared_error(y_test, predictions)
    print("Neural network MSE: " + str(mse))
    print("Neural network RMSE: " + str(np.sqrt(mse)))


def ex4():
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam

    # Read the data.
    PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
    CSV_DATA = "housing.data"
    df = pd.read_csv(PATH + CSV_DATA, header=None)

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head())
    print(df.tail())
    print(df.describe())

    # Convert DataFrame columns to vertical columns so they can be used by the NN.
    dataset = df.values
    X = dataset[:, 0:13]  # Columns 0 to 12
    y = dataset[:, 13]  # Columns 13
    ROW_DIM = 0
    COL_DIM = 1

    x_arrayReshaped = X.reshape(X.shape[ROW_DIM], X.shape[COL_DIM])
    y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)

    # Split the data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        x_arrayReshaped, y_arrayReshaped, test_size=0.3, random_state=0
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=0
    )

    # Define the model.
    def create_model():
        model = Sequential()
        model.add(
            Dense(13, input_dim=13, kernel_initializer="uniform", activation="relu")
        )
        model.add(Dense(1, kernel_initializer="uniform"))
        optimizer = Adam(lr=0.005)
        model.compile(loss="mean_squared_error", optimizer=optimizer)
        return model

    model = create_model()

    # Build the model.
    model = create_model()
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=10,
        verbose=1,
        validation_data=(X_val, y_val),
    )

    # Evaluate the model.
    predictions = model.predict(X_test)
    mse = metrics.mean_squared_error(y_test, predictions)
    print("Neural network MSE: " + str(mse))
    print("Neural network RMSE: " + str(np.sqrt(mse)))


def ex5():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
        classification_report,
    )

    PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
    FILE = "Social_Network_Ads.csv"
    data = pd.read_csv(PATH + FILE)
    y = data["Purchased"]
    X = data.copy()
    del X["User ID"]
    del X["Purchased"]
    X["Gender"] = X["Gender"].map({"Male": 0, "Female": 1})

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(data.head())

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Stochastic gradient descent models are sensitive to differences
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_trainScaled = scaler.transform(X_train)
    X_testScaled = scaler.transform(X_test)
    X_valScaled = scaler.transform(X_val)

    def showResults(networkStats):
        dfStats = pd.DataFrame.from_records(networkStats)
        dfStats = dfStats.sort_values(by=["f1"])
        print(dfStats)

    def evaluate_model(predictions, y_test):
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        print(
            "Precision: "
            + str(precision)
            + " "
            + "Recall: "
            + str(recall)
            + " "
            + "F1: "
            + str(f1)
        )
        return precision, recall, f1

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_trainScaled, y_train)
    predictions = clf.predict(X_testScaled)
    evaluate_model(predictions, y_test)

    COLUMN_DIMENSION = 1
    #######################################################################
    # Part 2
    from keras.models import Sequential
    from keras.layers import Dense

    # shape() obtains rows (dim=0) and columns (dim=1)
    n_features = X_trainScaled.shape[COLUMN_DIMENSION]

    def getPredictions(model, X_test):
        probabilities = model.predict(X_test)

        predictions = []
        for i in range(len(probabilities)):
            if probabilities[i][0] > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    ### Model parameters ############################
    batch_sizes = [10, 60, 100]
    epochList = [100, 200, 300]
    #################################################

    #######################################################################
    # Model building section.
    def create_model():
        model = Sequential()
        model.add(Dense(12, input_dim=n_features, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    networkStats = []
    for batch_size in batch_sizes:
        for epochs in epochList:
            model = create_model()
            history = model.fit(
                X_trainScaled,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                validation_data=(X_valScaled, y_val),
            )
            predictions = getPredictions(model, X_testScaled)

            precision, recall, f1 = evaluate_model(predictions, y_test)
            networkStats.append(
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "epochs": epochs,
                    "batch": batch_size,
                }
            )
    showResults(networkStats)
    #######################################################################


def ex6():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
        classification_report,
    )

    PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
    FILE = "Social_Network_Ads.csv"
    data = pd.read_csv(PATH + FILE)
    y = data["Purchased"]
    X = data.copy()
    del X["User ID"]
    del X["Purchased"]
    X["Gender"] = X["Gender"].map({"Male": 0, "Female": 1})

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(data.head())

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Stochastic gradient descent models are sensitive to differences
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_trainScaled = scaler.transform(X_train)
    X_testScaled = scaler.transform(X_test)
    X_valScaled = scaler.transform(X_val)

    def showResults(networkStats):
        dfStats = pd.DataFrame.from_records(networkStats)
        dfStats = dfStats.sort_values(by=["f1"])
        print(dfStats)

    def evaluate_model(predictions, y_test):
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        print(
            "Precision: "
            + str(precision)
            + " "
            + "Recall: "
            + str(recall)
            + " "
            + "F1: "
            + str(f1)
        )
        return precision, recall, f1

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_trainScaled, y_train)
    predictions = clf.predict(X_testScaled)
    evaluate_model(predictions, y_test)

    COLUMN_DIMENSION = 1
    #######################################################################
    # Part 2
    from keras.models import Sequential
    from keras.layers import Dense

    # shape() obtains rows (dim=0) and columns (dim=1)
    n_features = X_trainScaled.shape[COLUMN_DIMENSION]

    def getPredictions(model, X_test):
        probabilities = model.predict(X_test)

        predictions = []
        for i in range(len(probabilities)):
            if probabilities[i][0] > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    ### Model parameters ############################
    batch_sizes = [10, 30, 60, 100, 150, 200]
    epochList = [50, 100, 125, 150, 175, 200]
    #################################################

    #######################################################################
    # Model building section.
    def create_model():
        model = Sequential()
        model.add(Dense(12, input_dim=n_features, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    networkStats = []
    for batch_size in batch_sizes:
        for epochs in epochList:
            model = create_model()
            history = model.fit(
                X_trainScaled,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                validation_data=(X_valScaled, y_val),
            )
            predictions = getPredictions(model, X_testScaled)

            precision, recall, f1 = evaluate_model(predictions, y_test)
            networkStats.append(
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "epochs": epochs,
                    "batch": batch_size,
                }
            )
    showResults(networkStats)
    #######################################################################


def ex7():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
        classification_report,
    )

    PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
    FILE = "Social_Network_Ads.csv"
    data = pd.read_csv(PATH + FILE)
    y = data["Purchased"]
    X = data.copy()
    del X["User ID"]
    del X["Purchased"]
    X["Gender"] = X["Gender"].map({"Male": 0, "Female": 1})

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(data.head())

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Stochastic gradient descent models are sensitive to differences
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_trainScaled = scaler.transform(X_train)
    X_testScaled = scaler.transform(X_test)
    X_valScaled = scaler.transform(X_val)

    def showResults(networkStats):
        dfStats = pd.DataFrame.from_records(networkStats)
        dfStats = dfStats.sort_values(by=["f1"])
        print(dfStats)

    def evaluate_model(predictions, y_test):
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        print(
            "Precision: "
            + str(precision)
            + " "
            + "Recall: "
            + str(recall)
            + " "
            + "F1: "
            + str(f1)
        )
        return precision, recall, f1

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_trainScaled, y_train)
    predictions = clf.predict(X_testScaled)
    evaluate_model(predictions, y_test)

    COLUMN_DIMENSION = 1
    #######################################################################
    # Part 2
    from keras.models import Sequential
    from keras.layers import Dense

    # shape() obtains rows (dim=0) and columns (dim=1)
    n_features = X_trainScaled.shape[COLUMN_DIMENSION]

    def getPredictions(model, X_test):
        probabilities = model.predict(X_test)

        predictions = []
        for i in range(len(probabilities)):
            if probabilities[i][0] > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    ### Model parameters ############################
    neuronList = [5, 12, 25, 50, 100, 150]

    #################################################

    #######################################################################
    # Model building section.
    from keras.optimizers import RMSprop  # for adam optimizer

    def create_model(neuron):
        model = Sequential()
        model.add(Dense(neuron, input_dim=n_features, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        optimizer = RMSprop(learning_rate=0.001)
        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )
        return model

    networkStats = []
    EPOCHS = 200
    NUM_BATCHES = 60

    for neuron in neuronList:
        model = create_model(neuron)
        history = model.fit(
            X_trainScaled,
            y_train,
            epochs=EPOCHS,
            batch_size=NUM_BATCHES,
            verbose=1,
            validation_data=(X_valScaled, y_val),
        )
        predictions = getPredictions(model, X_testScaled)

        precision, recall, f1 = evaluate_model(predictions, y_test)
        networkStats.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "neurons": neuron,
            }
        )
    showResults(networkStats)
    #######################################################################


def main():
    # ex3()
    # ex4()
    # ex5()
    # ex6()
    ex7()


if __name__ == "__main__":
    main()
