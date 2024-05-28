def ex1():
    from sklearn.datasets import make_classification
    from torch import optim
    from skorch import NeuralNetClassifier
    import torch.nn as nn
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # This class could be any name.
    # nn.Module is needed to enable grid searching of parameters
    # with skorch later.
    class MyNeuralNet(nn.Module):
        # Define network objects.
        # Defaults are set for number of neurons and the
        # dropout rate.
        def __init__(self, num_neurons=10, dropout=0.1):
            super(MyNeuralNet, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer
            # feed forward network with n inputs and m output.
            self.dense0 = nn.Linear(20, num_neurons)
            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            self.dropout = nn.Dropout(dropout)

            # 2nd hidden layer.
            self.dense1 = nn.Linear(num_neurons, num_neurons)

            # 3rd hidden layer.
            self.dense2 = nn.Linear(num_neurons, num_neurons)

            # Output layer.
            self.output = nn.Linear(num_neurons, 2)

            # Softmax activation function allows for multiclass predictions.
            # In this case the prediction is binary.
            self.softmax = nn.Softmax(dim=-1)

        # Move data through the different network objects.
        def forward(self, x):
            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    def buildModel(x, y):
        # Trains the Neural Network with fixed hyperparameters
        # The Neural Net is initialized with fixed hyperparameters
        myNetwork = MyNeuralNet(num_neurons=20, dropout=0.1)
        nn = NeuralNetClassifier(
            myNetwork, max_epochs=21, lr=0.01, batch_size=12, optimizer=optim.RMSprop
        )
        model = nn.fit(x, y)
        return model

    def evaluateModel(model, X_test, y_test):
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    # Prep the data.
    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Build the model.
    model = buildModel(X_train, y_train)

    # Evaluate the model.
    evaluateModel(model, X_test, y_test)


def ex2():
    import torch
    import pandas as pd
    from torch import optim
    from skorch import NeuralNetClassifier
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # This class could be any name.
    # nn.Module is needed to enable grid searching of parameters
    # with skorch later.
    class MyNeuralNet(nn.Module):
        # Define network objects.
        # Defaults are set for number of neurons and the
        # dropout rate.
        def __init__(self, num_neurons=10, dropout=0.1):
            super(MyNeuralNet, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer
            # feed forward network with n inputs and m output.
            self.dense0 = nn.Linear(4, num_neurons)
            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            self.dropout = nn.Dropout(dropout)

            # 2nd hidden layer.
            self.dense1 = nn.Linear(num_neurons, num_neurons)

            # Output layer.
            self.output = nn.Linear(num_neurons, 2)

            # Softmax activation function allows for multiclass predictions.
            # In this case the prediction is binary.
            self.softmax = nn.Softmax(dim=-1)

        # Move data through the different network objects.
        def forward(self, x):
            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    def buildModel(x, y):
        # Trains the Neural Network with fixed hyperparameters
        # The Neural Net is initialized with fixed hyperparameters
        myNetwork = MyNeuralNet(num_neurons=10, dropout=0.1)
        nn = NeuralNetClassifier(
            myNetwork, max_epochs=10, lr=0.01, batch_size=12, optimizer=optim.RMSprop
        )
        model = nn.fit(x, y)
        return model

    def evaluateModel(model, X_test, y_test):
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
    df = pd.read_csv(PATH + "bill_authentication.csv")
    X = df.copy()
    del X["Class"]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    from sklearn.preprocessing import StandardScaler

    scalerX = StandardScaler()
    scaledXTrain = scalerX.fit_transform(X_train)
    scaledXTest = scalerX.transform(X_test)

    # Must convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(scaledXTrain, dtype=torch.float32)
    X_test_tensor = torch.tensor(scaledXTest, dtype=torch.float32)
    y_train_tensor = torch.tensor(list(y_train), dtype=torch.long)

    # Build the model.
    model = buildModel(X_train_tensor, y_train_tensor)

    # Evaluate the model.
    evaluateModel(model, X_test_tensor, y_test)


# Exercise 3, 4
def ex3():
    from sklearn.datasets import make_classification
    from torch import optim
    from skorch import NeuralNetClassifier
    import torch.nn as nn
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from skorch.callbacks import EpochScoring
    import matplotlib.pyplot as plt

    # This class could be any name.
    # nn.Module is needed to enable grid searching of parameters
    # with skorch later.
    class MyNeuralNet(nn.Module):
        # Define network objects.
        # Defaults are set for number of neurons and the
        # dropout rate.
        def __init__(self, num_neurons=10, dropout=0.1):
            super(MyNeuralNet, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer
            # feed forward network with n inputs and m output.
            self.dense0 = nn.Linear(20, num_neurons)
            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            self.dropout = nn.Dropout(dropout)

            # 2nd hidden layer.
            self.dense1 = nn.Linear(num_neurons, num_neurons)

            # Output layer.
            self.output = nn.Linear(num_neurons, 2)

            # Softmax activation function allows for multiclass predictions.
            # In this case the prediction is binary.
            self.softmax = nn.Softmax(dim=-1)

        # Move data through the different network objects.
        def forward(self, x):
            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    def buildModel(X_train, y_train):
        num_neurons = 25  # hidden layers
        net = NeuralNetClassifier(
            MyNeuralNet(num_neurons),
            max_epochs=200,
            lr=0.001,
            batch_size=100,
            optimizer=optim.RMSprop,
            callbacks=[
                EpochScoring(scoring="accuracy", name="towa_train_acc", on_train=True)
            ],
        )
        # Pipeline execution
        model = net.fit(X_train, y_train)
        return model, net

    def evaluateModel(model, X_test, y_test):
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    # Prep the data.
    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Build the model.
    model, net = buildModel(X_train, y_train)

    # Evaluate the model.
    evaluateModel(model, X_test, y_test)

    plt.rcParams.update({"font.size": 30})

    def drawLossPlot(net):
        plt.plot(net.history[:, "train_loss"], color="blue", label="train")
        plt.plot(net.history[:, "valid_loss"], color="orange", label="val")
        plt.legend()
        plt.show()

    def drawAccuracyPlot(net):
        plt.plot(net.history[:, "towa_train_acc"], color="blue", label="train")
        plt.plot(net.history[:, "valid_acc"], color="orange", label="val")
        plt.legend()
        plt.show()

    drawLossPlot(net)
    drawAccuracyPlot(net)


def ex5():
    from torch import optim
    from skorch import NeuralNetClassifier
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # This class could be any name.
    # nn.Module is needed to enable grid searching of parameters
    # with skorch later.
    class MyNeuralNet(nn.Module):
        # Define network objects.
        # Defaults are set for number of neurons and the
        # dropout rate.
        def __init__(self, num_neurons=10, dropout=0.1):
            super(MyNeuralNet, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer
            # feed forward network with n inputs and m output.
            self.dense0 = nn.Linear(3, num_neurons)
            print("Dense layer type:")
            print(self.dense0.weight.dtype)

            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            self.dropout = nn.Dropout(dropout)

            # 2nd hidden layer.
            self.dense1 = nn.Linear(num_neurons, num_neurons)

            # Output layer.
            self.output = nn.Linear(num_neurons, 2)

            # Softmax activation function allows for multiclass predictions.
            # In this case the prediction is binary.
            self.softmax = nn.Softmax(dim=-1)

        # Move data through the different network objects.
        def forward(self, x):
            x = x.to(torch.float32)
            print("X type: ")
            print(x.dtype)

            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    def buildModel(x, y):
        # Through a grid search, the optimal hyperparameters are found
        # A pipeline is used in order to scale and train the neural net
        # The grid search module from scikit-learn wraps the pipeline

        # The Neural Net is instantiated, none hyperparameter is provided
        nn = NeuralNetClassifier(MyNeuralNet, verbose=0, train_split=False)
        # The pipeline is instantiated, it wraps scaling and training phase
        pipeline = Pipeline([("nn", nn)])

        # The parameters for the grid search are defined
        # Must use prefix "nn__" when setting hyperparamters for the training phase
        # Must use prefix "nn__module__" when setting hyperparameters for the Neural Net
        params = {
            "nn__max_epochs": [10, 20],
            "nn__lr": [0.1, 0.01],
            "nn__module__num_neurons": [5, 10],
            "nn__module__dropout": [0.1, 0.5],
            "nn__optimizer": [optim.Adam, optim.SGD, optim.RMSprop],
        }

        # The grid search module is instantiated
        gs = GridSearchCV(
            pipeline, params, refit=True, cv=3, scoring="balanced_accuracy", verbose=1
        )

        return gs.fit(x, y)

    def evaluateModel(model, X_test, y_test):
        print("Towa's network model for college admissions.")
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    ######################### Data Prep Start
    # Setup data.
    import pandas as pd
    import numpy as np
    import torch

    candidates = {
        "gmat": [
            780,
            750,
            690,
            710,
            680,
            730,
            690,
            720,
            740,
            690,
            610,
            690,
            710,
            680,
            770,
            610,
            580,
            650,
            540,
            590,
            620,
            600,
            550,
            550,
            570,
            670,
            660,
            580,
            650,
            660,
            640,
            620,
            660,
            660,
            680,
            650,
            670,
            580,
            590,
            690,
        ],
        "gpa": [
            4,
            3.9,
            3.3,
            3.7,
            3.9,
            3.7,
            2.3,
            3.3,
            3.3,
            1.7,
            2.7,
            3.7,
            3.7,
            3.3,
            3.3,
            3,
            2.7,
            3.7,
            2.7,
            2.3,
            3.3,
            2,
            2.3,
            2.7,
            3,
            3.3,
            3.7,
            2.3,
            3.7,
            3.3,
            3,
            2.7,
            4,
            3.3,
            3.3,
            2.3,
            2.7,
            3.3,
            1.7,
            3.7,
        ],
        "work_experience": [
            3,
            4,
            3,
            5,
            4,
            6,
            1,
            4,
            5,
            1,
            3,
            5,
            6,
            4,
            3,
            1,
            4,
            6,
            2,
            3,
            2,
            1,
            4,
            1,
            2,
            6,
            4,
            2,
            6,
            5,
            1,
            2,
            4,
            6,
            5,
            1,
            2,
            1,
            4,
            5,
        ],
        "admitted": [
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
        ],
    }

    df = pd.DataFrame(
        candidates, columns=["gmat", "gpa", "work_experience", "admitted"]
    )
    y = np.array(df["admitted"])
    X = df.copy()
    del X["admitted"]
    X = X

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # define standard scaler
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # transform data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)  # whole number needed
    y_test = torch.tensor(y_test, dtype=torch.long)  # for classification.
    ######################### End

    # Build the model.
    model = buildModel(X_train, y_train)

    print("Best parameters:")
    print(model.best_params_)

    # Evaluate the model.
    evaluateModel(model.best_estimator_, X_test, y_test)


def ex6():
    from torch import optim
    from skorch import NeuralNetClassifier
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # This class could be any name.
    # nn.Module is needed to enable grid searching of parameters
    # with skorch later.
    class MyNeuralNet(nn.Module):
        # Define network objects.
        # Defaults are set for number of neurons and the
        # dropout rate.
        def __init__(self, num_neurons=10, dropout=0.1):
            super(MyNeuralNet, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer
            # feed forward network with n inputs and m output.
            self.dense0 = nn.Linear(2, num_neurons)
            print("Dense layer type:")
            print(self.dense0.weight.dtype)

            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            self.dropout = nn.Dropout(dropout)

            # 2nd hidden layer.
            self.dense1 = nn.Linear(num_neurons, num_neurons)

            # Output layer.
            self.output = nn.Linear(num_neurons, 2)

            # Softmax activation function allows for multiclass predictions.
            # In this case the prediction is binary.
            self.softmax = nn.Softmax(dim=-1)

        # Move data through the different network objects.
        def forward(self, x):
            x = x.to(torch.float32)
            print("X type: ")
            print(x.dtype)

            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    def buildModel(x, y):
        # Through a grid search, the optimal hyperparameters are found
        # A pipeline is used in order to scale and train the neural net
        # The grid search module from scikit-learn wraps the pipeline

        # The Neural Net is instantiated, none hyperparameter is provided
        nn = NeuralNetClassifier(MyNeuralNet, verbose=0, train_split=False)
        # The pipeline is instantiated, it wraps scaling and training phase
        pipeline = Pipeline([("nn", nn)])

        # The parameters for the grid search are defined
        # Must use prefix "nn__" when setting hyperparamters for the training phase
        # Must use prefix "nn__module__" when setting hyperparameters for the Neural Net
        params = {
            "nn__max_epochs": [10, 20, 30],
            "nn__lr": [0.1, 0.01],
            "nn__module__num_neurons": [5, 7, 10],
            "nn__module__dropout": [0.05, 0.1, 0.5],
            "nn__optimizer": [optim.Adam, optim.SGD, optim.RMSprop],
        }

        # The grid search module is instantiated
        gs = GridSearchCV(
            pipeline, params, refit=True, cv=3, scoring="balanced_accuracy", verbose=1
        )

        return gs.fit(x, y)

    def evaluateModel(model, X_test, y_test):
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    ######################### Data Prep Start
    # Setup data.
    import pandas as pd
    import numpy as np
    import torch

    PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
    df = pd.read_csv(PATH + "fluDiagnosis.csv")
    X = df.copy()
    del X["Diagnosed"]
    y = np.array(df["Diagnosed"])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # define standard scaler
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # transform data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)  # whole number needed
    y_test = torch.tensor(y_test, dtype=torch.long)  # for classification.
    ######################### End

    # Build the model.
    model = buildModel(X_train, y_train)

    print("Towa's flu diagnosis network best parameters:")
    print(model.best_params_)

    # Evaluate the model.
    evaluateModel(model.best_estimator_, X_test, y_test)


def ex7():
    from torch import optim
    from skorch import NeuralNetClassifier
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from skorch.callbacks import EpochScoring
    import matplotlib.pyplot as plt

    # This class could be any name.
    # nn.Module is needed to enable grid searching of parameters
    # with skorch later.
    class MyNeuralNet(nn.Module):
        # Define network objects.
        # Defaults are set for number of neurons and the
        # dropout rate.
        def __init__(self, num_neurons=10, dropout=0.1):
            super(MyNeuralNet, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer
            # feed forward network with n inputs and m output.
            self.dense0 = nn.Linear(2, num_neurons)
            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            self.dropout = nn.Dropout(dropout)

            # 2nd hidden layer.
            self.dense1 = nn.Linear(num_neurons, num_neurons)

            # Output layer.
            self.output = nn.Linear(num_neurons, 2)

            # Softmax activation function allows for multiclass predictions.
            # In this case the prediction is binary.
            self.softmax = nn.Softmax(dim=-1)

        # Move data through the different network objects.
        def forward(self, x):
            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    def buildModel(X_train, y_train):
        num_neurons = 10  # hidden layers
        net = NeuralNetClassifier(
            MyNeuralNet(num_neurons, 0.5),
            max_epochs=10,
            lr=0.1,
            batch_size=100,
            optimizer=optim.Adam,
            callbacks=[
                EpochScoring(scoring="accuracy", name="train_acc", on_train=True)
            ],
        )
        # Pipeline execution
        model = net.fit(X_train, y_train)
        return model, net

    def evaluateModel(model, X_test, y_test):
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    ######################### Data Prep Start
    # Setup data.
    import pandas as pd
    import numpy as np
    import torch

    PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
    df = pd.read_csv(PATH + "fluDiagnosis.csv")
    X = df.copy()
    del X["Diagnosed"]
    y = np.array(df["Diagnosed"])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # define standard scaler
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # transform data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)  # whole number needed
    y_test = torch.tensor(y_test, dtype=torch.long)  # for classification.
    ######################### End

    # Build the model.
    model, net = buildModel(X_train, y_train)

    # Evaluate the model.
    evaluateModel(model, X_test, y_test)

    plt.rcParams.update({"font.size": 30})

    def drawLossPlot(net):
        plt.plot(net.history[:, "train_loss"], color="blue", label="train")
        plt.plot(net.history[:, "valid_loss"], color="orange", label="val")
        plt.legend()
        plt.show()

    def drawAccuracyPlot(net):
        plt.plot(net.history[:, "train_acc"], color="blue", label="train")
        plt.plot(net.history[:, "valid_acc"], color="orange", label="val")
        plt.legend()
        plt.show()

    drawLossPlot(net)
    drawAccuracyPlot(net)


def ex9():
    import torch.nn as nn
    import torch
    from sklearn.datasets import load_iris
    from torch import optim
    from skorch import NeuralNetClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    from skorch.callbacks import EpochScoring

    # Get iris data.
    iris = load_iris()
    X = iris.data
    Y = iris.target

    # Split and scale the data.
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)  # whole number needed

    class Net(nn.Module):
        def __init__(self, num_features, num_neurons, output_dim):
            super(Net, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer of a
            # feed forward network with n inputs and m output.
            self.dense0 = nn.Linear(num_features, num_neurons)
            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            DROPOUT = 0.1
            self.dropout = nn.Dropout(DROPOUT)

            # 2nd hidden layer.
            self.dense1 = nn.Linear(num_neurons, output_dim)

            # Output layer.
            self.output = nn.Linear(output_dim, 3)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    def evaluateModel(model, X_test, y_test):
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    def buildModel(X_train, y_train):
        input_dim = 4  # how many Variables are in the dataset
        num_neurons = 25  # hidden layers
        output_dim = 3  # number of classes

        net = NeuralNetClassifier(
            Net(input_dim, num_neurons, output_dim),
            max_epochs=200,
            lr=0.001,
            batch_size=100,
            optimizer=optim.RMSprop,
            callbacks=[
                EpochScoring(scoring="accuracy", name="train_acc", on_train=True)
            ],
        )
        model = net.fit(X_train, y_train)
        return model, net

    model, net = buildModel(X_train, y_train)
    evaluateModel(model, X_test, y_test)
    print("Done")

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 30})

    def drawLossPlot(net):
        plt.plot(net.history[:, "train_loss"], color="blue", label="train")
        plt.plot(net.history[:, "valid_loss"], color="orange", label="val")
        plt.legend()
        plt.show()

    def drawAccuracyPlot(net):
        plt.plot(net.history[:, "train_acc"], color="blue", label="train")
        plt.plot(net.history[:, "valid_acc"], color="orange", label="val")
        plt.legend()
        plt.show()

    drawLossPlot(net)
    drawAccuracyPlot(net)


def ex10():
    import torch.nn as nn
    import torch
    from sklearn.datasets import load_iris
    from torch import optim
    from skorch import NeuralNetClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    from skorch.callbacks import EpochScoring
    from skorch.callbacks import EarlyStopping

    # Get iris data.
    iris = load_iris()
    X = iris.data
    Y = iris.target

    # Split and scale the data.
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)  # whole number needed

    class Net(nn.Module):
        def __init__(self, num_features, num_neurons, output_dim):
            super(Net, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer of a
            # feed forward network with n inputs and m output.
            self.dense0 = nn.Linear(num_features, num_neurons)
            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            DROPOUT = 0.1
            self.dropout = nn.Dropout(DROPOUT)

            # 2nd hidden layer.
            self.dense1 = nn.Linear(num_neurons, output_dim)

            # Output layer.
            self.output = nn.Linear(output_dim, 3)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    def evaluateModel(model, X_test, y_test):
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    def buildModel(X_train, y_train):
        input_dim = 4  # how many Variables are in the dataset
        num_neurons = 25  # hidden layers
        output_dim = 3  # number of classes

        net = NeuralNetClassifier(
            Net(input_dim, num_neurons, output_dim),
            max_epochs=1000,
            lr=0.001,
            batch_size=100,
            optimizer=optim.RMSprop,
            callbacks=[
                EpochScoring(scoring="accuracy", name="train_acc", on_train=True),
                EarlyStopping(patience=100),
            ],
        )
        model = net.fit(X_train, y_train)
        return model, net

    model, net = buildModel(X_train, y_train)
    evaluateModel(model, X_test, y_test)
    print("Done")

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 30})

    def drawLossPlot(net):
        plt.plot(net.history[:, "train_loss"], color="blue", label="train")
        plt.plot(net.history[:, "valid_loss"], color="orange", label="val")
        plt.legend()
        plt.show()

    def drawAccuracyPlot(net):
        plt.plot(net.history[:, "train_acc"], color="blue", label="train")
        plt.plot(net.history[:, "valid_acc"], color="orange", label="val")
        plt.legend()
        plt.show()

    drawLossPlot(net)
    drawAccuracyPlot(net)


def ex11():
    import pandas as pd
    from torch import optim
    from skorch import NeuralNetClassifier
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from skorch.callbacks import EpochScoring
    import matplotlib.pyplot as plt
    import torch
    from sklearn.preprocessing import StandardScaler
    from skorch.callbacks import EarlyStopping

    # This class could be any name.
    # nn.Module is needed to enable grid searching of parameters
    # with skorch later.
    class MyNeuralNet(nn.Module):
        # Define network objects.
        # Defaults are set for number of neurons and the
        # dropout rate.
        def __init__(self, num_neurons=10, dropout=0.1):
            super(MyNeuralNet, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer
            # feed forward network with n inputs and m output.
            self.dense0 = nn.Linear(28, num_neurons)
            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            self.dropout = nn.Dropout(dropout)

            # 2nd hidden layer.
            self.dense1 = nn.Linear(num_neurons, num_neurons)

            # 3rd hidden layer.
            self.dense2 = nn.Linear(num_neurons, num_neurons)

            # Output layer.
            self.output = nn.Linear(num_neurons, 4)

            # Softmax activation function allows for multiclass predictions.
            # In this case the prediction is binary.
            self.softmax = nn.Softmax(dim=-1)

        # Move data through the different network objects.
        def forward(self, x):
            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    def buildModel(X_train, y_train):
        num_neurons = 25  # hidden layers
        net = NeuralNetClassifier(
            MyNeuralNet(num_neurons),
            max_epochs=1000,
            lr=0.001,
            batch_size=100,
            optimizer=optim.RMSprop,
            callbacks=[
                EpochScoring(scoring="accuracy", name="train_acc", on_train=True),
                EarlyStopping(patience=100),
            ],
        )
        # Pipeline execution
        model = net.fit(X_train, y_train)
        return model, net

    def evaluateModel(model, X_test, y_test):
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    def getCustomerSegmentationData():
        PATH = "/Users/elber/Documents/COMP 4948 - Datasets/"
        df = pd.read_csv(PATH + "CustomerSegmentation.csv")
        df = pd.get_dummies(
            df,
            columns=[
                "Gender",
                "Ever_Married",
                "Graduated",
                "Profession",
                "Spending_Score",
                "Var_1",
            ],
        )
        df["Segmentation"] = df["Segmentation"].replace(
            {"A": 0, "B": 1, "C": 2, "D": 3}
        )
        print(df["Segmentation"].value_counts())
        X = df.copy()
        del X["Segmentation"]
        y = df["Segmentation"]
        return X, y

    X, y = getCustomerSegmentationData()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)  # whole number needed

    # Build the model.
    model, net = buildModel(X_train, y_train)

    # Evaluate the model.
    evaluateModel(model, X_test, y_test)

    plt.rcParams.update({"font.size": 30})

    def drawLossPlot(net):
        plt.plot(net.history[:, "train_loss"], color="blue", label="train")
        plt.plot(net.history[:, "valid_loss"], color="orange", label="val")
        plt.legend()
        plt.show()

    def drawAccuracyPlot(net):
        plt.plot(net.history[:, "train_acc"], color="blue", label="train")
        plt.plot(net.history[:, "valid_acc"], color="orange", label="val")
        plt.legend()
        plt.show()

    drawLossPlot(net)
    drawAccuracyPlot(net)


def main():
    # ex1()
    # ex2()
    # ex3()
    # ex5()
    # ex6()
    # ex7()
    # ex9()
    # ex10()
    ex11()


if __name__ == "__main__":
    main()
