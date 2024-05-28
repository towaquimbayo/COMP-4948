# Exercise 1, 2
def ex1():
    # Creating an overfit situation with the moons data set.
    from sklearn.datasets import make_moons
    from keras.layers import Dense
    from keras.models import Sequential
    import matplotlib.pyplot as plt

    # Generate 2d classification dataset.
    X, y = make_moons(n_samples=100, noise=0.2, random_state=1)

    # Split data into train and test.
    n_train = 30
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    # Define the model.
    model = Sequential()
    model.add(Dense(500, input_dim=2, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Fit the model.
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ModelCheckpoint
    from keras.models import load_model

    # simple early stopping
    # patience:  # of epochs observed where no improvement before exiting.
    # mode:      Could be max, min, or auto.
    # min_delta: Amount of change needed to be considered an improvement.
    es = EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, min_delta=0.000001, patience=200
    )
    mc = ModelCheckpoint(
        "best_model.h5", monitor="val_loss", mode="min", verbose=1, save_best_only=True
    )

    # fit model
    history = model.fit(
        trainX,
        trainy,
        validation_data=(testX, testy),
        epochs=4000,
        verbose=0,
        callbacks=[es, mc],
    )

    # load the saved model
    model = load_model("best_model.h5")

    # Evaluate the model.
    train_loss, train_acc = model.evaluate(trainX, trainy, verbose=0)
    test_loss, test_acc = model.evaluate(testX, testy, verbose=0)
    print("Train accuracy: %.3f, Test accuracy: %.3f" % (train_acc, test_acc))
    print("Train loss: %.3f, Test loss: %.3f" % (train_loss, test_loss))

    # Plot loss learning curves.
    plt.subplot(211)
    plt.title("Cross-Entropy Loss", pad=-40)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="test")
    plt.legend()

    # Plot accuracy learning curves.
    plt.subplot(212)
    plt.title("Accuracy", pad=-40)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="test")
    plt.legend()
    plt.show()


def ex3():
    # Creating an overfit situation with the moons data set.
    from sklearn.datasets import make_moons
    from keras.layers import Dense
    from keras.models import Sequential
    import matplotlib.pyplot as plt

    # Generate 2d classification dataset.
    X, y = make_moons(n_samples=100, noise=0.2, random_state=1)

    # Split data into train and test.
    n_train = 30
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    # Define the model.
    model = Sequential()
    model.add(Dense(500, input_dim=2, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Fit the model.
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ModelCheckpoint
    from keras.models import load_model

    # simple early stopping
    # patience:  # of epochs observed where no improvement before exiting.
    # mode:      Could be max, min, or auto.
    # min_delta: Amount of change needed to be considered an improvement.
    es = EarlyStopping(
        monitor="val_accuracy", mode="max", verbose=1, min_delta=0.000001, patience=200
    )
    mc = ModelCheckpoint(
        "best_model.h5",
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )

    # fit model
    history = model.fit(
        trainX,
        trainy,
        validation_data=(testX, testy),
        epochs=4000,
        verbose=0,
        callbacks=[es, mc],
    )

    # load the saved model
    model = load_model("best_model.h5")

    # Evaluate the model.
    train_loss, train_acc = model.evaluate(trainX, trainy, verbose=0)
    test_loss, test_acc = model.evaluate(testX, testy, verbose=0)
    print("Train accuracy: %.3f, Test accuracy: %.3f" % (train_acc, test_acc))
    print("Train loss: %.3f, Test loss: %.3f" % (train_loss, test_loss))

    # Plot loss learning curves.
    plt.subplot(211)
    plt.title("Cross-Entropy Loss", pad=-40)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="test")
    plt.legend()

    # Plot accuracy learning curves.
    plt.subplot(212)
    plt.title("Accuracy", pad=-40)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="test")
    plt.legend()
    plt.show()


def ex4():
    # mlp with unscaled data for the regression problem
    from sklearn.datasets import make_regression
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import SGD
    import matplotlib.pyplot as plt

    # Generate the regression dataset.
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)

    plt.hist(y)
    plt.title("Unscaled Input")
    plt.show()

    # Split into train and test.
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    clipResults = []

    def buildModel(clipSize):
        # Define the model.
        model = Sequential()
        model.add(
            Dense(25, input_dim=20, activation="relu", kernel_initializer="he_uniform")
        )
        model.add(Dense(1, activation="linear"))

        # Compile the model.
        opt = SGD(lr=0.01, momentum=0.9, clipnorm=clipSize)
        model.compile(loss="mean_squared_error", optimizer=opt)

        # Fit the model.
        history = model.fit(
            trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=1
        )

        # Evaluate the model.
        train_mse = model.evaluate(trainX, trainy, verbose=0)
        test_mse = model.evaluate(testX, testy, verbose=0)
        print("Train MSE: %.3f, Test MSE: %.3f" % (train_mse, test_mse))
        clipResults.append(
            {"train mse": train_mse, "test mse": test_mse, "clip size": clipSize}
        )
        # Plot losses during training.
        plt.title("Losses")
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="test")
        plt.legend()
        plt.show()

    clipSizes = [0.5, 0.9, 1, 1.1, 1.2, 1.5, 2]
    for i in range(0, len(clipSizes)):
        buildModel(clipSizes[i])

    for clipResult in clipResults:
        print(clipResult)


def ex5():
    # mlp with unscaled data for the regression problem
    from sklearn.datasets import make_regression
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import SGD
    import matplotlib.pyplot as plt

    # Generate the regression dataset.
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)

    plt.hist(y)
    plt.title("Unscaled Input")
    plt.show()

    # Split into train and test.
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    clipResults = []

    def buildModel(clipSize):
        # Define the model.
        model = Sequential()
        model.add(
            Dense(25, input_dim=20, activation="relu", kernel_initializer="he_uniform")
        )
        model.add(Dense(1, activation="linear"))

        # Compile the model.
        opt = SGD(lr=0.01, momentum=0.9, clipnorm=clipSize, clipvalue=5.0)
        model.compile(loss="mean_squared_error", optimizer=opt)

        # Fit the model.
        history = model.fit(
            trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=1
        )

        # Evaluate the model.
        train_mse = model.evaluate(trainX, trainy, verbose=0)
        test_mse = model.evaluate(testX, testy, verbose=0)
        print("Train MSE: %.3f, Test MSE: %.3f" % (train_mse, test_mse))
        clipResults.append(
            {"train mse": train_mse, "test mse": test_mse, "clip size": clipSize}
        )
        # Plot losses during training.
        plt.title("Losses")
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="test")
        plt.legend()
        plt.show()

    clipSizes = [0.5, 0.9, 1, 1.1, 1.2, 1.5, 2]
    for i in range(0, len(clipSizes)):
        buildModel(clipSizes[i])

    for clipResult in clipResults:
        print(clipResult)


# Exercise 6, 7, 8, 9, 10, 11
def ex6():
    from sklearn.datasets import make_blobs
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import SGD
    from keras.utils import to_categorical
    import matplotlib.pyplot as plt

    # Generate the data.
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    import pandas as pd

    def prepare_data():
        PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
        # load the dataset
        df = pd.read_csv(PATH + "diabetes.csv", sep=",")

        # split into input (X) and output (y) variables
        X = df[
            [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ]
        ]
        y = df[["Outcome"]]

        # Split into train and test data sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        return X_train, X_test, y_train, y_test

    # Build the base model.
    def get_base_model(trainX, trainy):
        # define the keras model
        model = Sequential()
        model.add(
            Dense(230, input_dim=8, activation="relu", kernel_initializer="he_normal")
        )

        model.add(Dense(1, activation="sigmoid"))
        opitimizer = SGD(
            learning_rate=0.0005,
            momentum=0.9,
            name="SGD",
        )

        # Compile the keras model.
        model.compile(
            loss="binary_crossentropy", optimizer=opitimizer, metrics=["accuracy"]
        )

        # Fit the keras model on the dataset.
        model.fit(trainX, trainy, epochs=200, batch_size=10)
        return model

    stats = []

    # Evaluate the model.
    def evaluate_model(numLayers, model, trainX, testX, trainy, testy):
        train_loss, train_acc = model.evaluate(trainX, trainy, verbose=1)
        test_loss, test_acc = model.evaluate(testX, testy, verbose=1)
        stats.append(
            {
                "# layers": numLayers,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_loss": train_loss,
                "test_loss": test_loss,
            }
        )

    # Add one new layer and re-train only the new layer.
    def add_layer(model, trainX, trainy):
        # Store the output layer.
        output_layer = model.layers[-1]

        # Remove the output layer.
        model.pop()

        # Mark all remaining layers as non-trainable.
        for layer in model.layers:
            layer.trainable = False

        # Add a new hidden layer.
        model.add(Dense(230, activation="relu", kernel_initializer="he_uniform"))

        # Add the output layer back.
        model.add(output_layer)

        # fit model
        model.fit(trainX, trainy, epochs=300, verbose=1)
        return model

    # Get the data and build the base model.
    trainX, testX, trainy, testy = prepare_data()
    model = get_base_model(trainX, trainy)

    # Evaluate the base model
    scores = dict()
    evaluate_model(-1, model, trainX, testX, trainy, testy)

    # add layers and evaluate the updated model
    n_layers = 14
    for i in range(n_layers):
        model = add_layer(model, trainX, trainy)
        evaluate_model(i, model, trainX, testX, trainy, testy)

    columns = ["# layers", "train_acc", "test_acc", "train_loss", "test_loss"]

    df = pd.DataFrame(columns=columns)
    for i in range(0, len(stats)):
        df = df._append(stats[i], ignore_index=True)
    print(df)


def main():
    # ex1()
    # ex3()
    # ex4()
    # ex5()
    ex6()


if __name__ == "__main__":
    main()
