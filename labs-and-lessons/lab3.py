# Exercise 4, 5, 6, 7, 8, 9
def ex4():
    # importing libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    ### DATA           #######################################
    # input data
    # (no scaling required in this case for simplicity)
    X_train_scaled = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    # output data
    y_train = np.array([[0], [1], [1], [0]])
    ### DATA           #######################################

    def plotSigmoid(x, y):
        plt.plot(x, y)
        plt.title("Sigmoid Function")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    # defining sigmoid activation function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # defining derivative of sigmoid activation function
    def sigmoid_derivative(x):
        return x * (1 - x)

    # Pass data through layers.
    # Each layer sums applied weights and bias, passes to
    # an activation function.
    def feedForward(X_scaled):
        sum1 = np.dot(X_scaled, weights_0_1) + bias_0_1
        sum1 = np.array(sum1)
        layer_1 = sigmoid(sum1)  # Activation function.
        plotSigmoid(sum1, layer_1)  # TODO: PLOT SIGMOID
        sum2 = np.dot(layer_1, weights_1_2) + bias_1_2
        layer_2 = sigmoid(sum2)  # Activation function.
        return layer_1, layer_2

    # Update weights and bias from front to back.
    def backPropogate(weights_0_1, bias_0_1, weights_1_2, bias_1_2):
        yDf = pd.DataFrame(data=y_train, columns=["admitted"])

        # Calculate prediction error.
        layer_2_error = layer_2 - np.array(yDf["admitted"]).reshape(-1, 1)
        # Get rate of change of cost function.
        layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)

        # Determine layer 1 error as cost rate of change * layer 2 weights.
        layer_1_error = layer_2_delta.dot(weights_1_2.T)
        # Get rate of change for layer 1.
        layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

        # Update weights and bias.
        weights_1_2 -= layer_1.T.dot(layer_2_delta) * learning_rate
        weights_0_1 -= layer_0.T.dot(layer_1_delta) * learning_rate

        bias_1_2 -= np.sum(layer_2_delta, axis=0, keepdims=True) * learning_rate
        bias_0_1 -= np.sum(layer_1_delta, axis=0, keepdims=True) * learning_rate
        return weights_0_1, bias_0_1, weights_1_2, bias_1_2

    # defining learning rate
    learning_rate = 0.1

    # These weights would normally be generated randomly
    # or with kernel initializers.
    weights_0_1 = np.array(
        [
            [0.10473281, 0.23991864, 0.51106061, 0.97739018],
            [0.46591006, 0.54318817, 0.58782883, 0.68117129],
            [0.0502301, 0.22142866, 0.86126238, 0.72482657],
        ]
    )
    bias_0_1 = np.array([[0.15025418, 0.73481849, 0.90219478, 0.30605943]])
    weights_1_2 = np.array([[0.62996657], [0.25984049], [0.72180012], [0.81730325]])
    bias_1_2 = np.array([[0.09842751]])

    # training loop
    EPOCHS = 10000
    for i in range(EPOCHS):
        layer_0 = X_train_scaled
        # Feed data forward.
        layer_1, layer_2 = feedForward(X_train_scaled)

        # Back propagate to update weights and bias.
        weights_0_1, bias_0_1, weights_1_2, bias_1_2 = backPropogate(
            weights_0_1, bias_0_1, weights_1_2, bias_1_2
        )

        # printing output
        print("\nEpoch:", i + 1)
        print("Actual data:")
        print(y_train)
        print("ANN predictions:")
        print(layer_2)


def ex10():
    # importing libraries
    import numpy as np
    import pandas as pd

    ### DATA           #######################################
    # Setup data.
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

    from sklearn.model_selection import train_test_split

    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # transform data
    X_train_scaled = X_train  # No scaling
    ### DATA           #######################################

    # defining sigmoid activation function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # defining derivative of sigmoid activation function
    def sigmoid_derivative(x):
        return x * (1 - x)

    # Pass data through layers.
    # Each layer sums applied weights and bias, passes to
    # an activation function.
    def feedForward(X_scaled):
        sum1 = np.dot(X_scaled, weights_0_1) + bias_0_1
        sum1 = np.array(sum1)
        layer_1 = sigmoid(sum1)  # Activation function.
        sum2 = np.dot(layer_1, weights_1_2) + bias_1_2
        layer_2 = sigmoid(sum2)  # Activation function.
        return layer_1, layer_2

    # Update weights and bias from front to back.
    def backPropogate(weights_0_1, bias_0_1, weights_1_2, bias_1_2):
        yDf = pd.DataFrame(data=y_train, columns=["admitted"])

        # Calculate prediction error.
        layer_2_error = layer_2 - np.array(yDf["admitted"]).reshape(-1, 1)
        # Get rate of change of cost function.
        layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)

        # Determine layer 1 error as cost rate of change * layer 2 weights.
        layer_1_error = layer_2_delta.dot(weights_1_2.T)
        # Get rate of change for layer 1.
        layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

        # Update weights and bias.
        weights_1_2 -= layer_1.T.dot(layer_2_delta) * learning_rate
        weights_0_1 -= layer_0.T.dot(layer_1_delta) * learning_rate

        bias_1_2 -= np.sum(layer_2_delta, axis=0, keepdims=True) * learning_rate
        bias_0_1 -= np.sum(layer_1_delta, axis=0, keepdims=True) * learning_rate
        return weights_0_1, bias_0_1, weights_1_2, bias_1_2

    # defining learning rate
    learning_rate = 0.1

    # These weights would normally be generated randomly
    # or with kernel initializers.
    weights_0_1 = np.array(
        [
            [0.10473281, 0.23991864, 0.51106061, 0.97739018],
            [0.46591006, 0.54318817, 0.58782883, 0.68117129],
            [0.0502301, 0.22142866, 0.86126238, 0.72482657],
        ]
    )
    bias_0_1 = np.array([[0.15025418, 0.73481849, 0.90219478, 0.30605943]])
    weights_1_2 = np.array([[0.62996657], [0.25984049], [0.72180012], [0.81730325]])
    bias_1_2 = np.array([[0.09842751]])

    # training loop
    EPOCHS = 10000
    for i in range(EPOCHS):
        layer_0 = X_train_scaled
        # Feed data forward.
        layer_1, layer_2 = feedForward(X_train_scaled)

        # Back propagate to update weights and bias.
        weights_0_1, bias_0_1, weights_1_2, bias_1_2 = backPropogate(
            weights_0_1, bias_0_1, weights_1_2, bias_1_2
        )

        # printing output
        print("\nEpoch:", i + 1)
        print("Actual data:")
        print(y_train)
        print("ANN predictions:")
        print(layer_2)


def ex11():
    # importing libraries
    import numpy as np
    import pandas as pd

    ### DATA           #######################################
    # Setup data.
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

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    # define standard scaler
    scaler = StandardScaler()

    # transform data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ##########################################################

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense

    # Build a network model of sequential layers.
    model = Sequential()

    NUM_COLS = 3
    # Add 1st hidden layer. Note 1st hidden layer also receives data.
    # The input array must contain two feature columns and any number of rows.
    model.add(Dense(10, activation="sigmoid", input_shape=(NUM_COLS,)))

    # Add 2nd hidden layer.
    model.add(Dense(3, activation="sigmoid"))

    # Add output layer.
    model.add(Dense(1, activation="sigmoid"))

    # Compile the model.
    # Binary cross entropy is used to measure error cost for binary predictions.
    model.compile(loss="binary_crossentropy", metrics=["accuracy"])

    # Fit the model.
    # An epoch is one iteration for all samples through the network.
    # verbose can be set to 1 to show detailed output during training.
    model.fit(X_train_scaled, y_train, epochs=2000, verbose=1)

    # Evaluate the model.
    loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print("Test Accuracy: %.3f" % acc)

    # Make predictions.
    yhats = model.predict(X_test_scaled)
    print("Actual:")
    print(y_test)
    print("Predicted: ")
    print(yhats)
    predictions = []

    from sklearn.metrics import classification_report

    def showClassificationReport(y_test, yhats):
        # Convert continous predictions to
        # 0 or 1.
        for i in range(0, len(yhats)):
            if yhats[i] > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        print(classification_report(y_test, predictions))

    showClassificationReport(y_test, yhats)


def ex13():
    # importing libraries
    import numpy as np
    import pandas as pd

    ### DATA           #######################################
    # Setup data.
    # Load the flower feature data into a DataFrame.
    data = [
        {"Length": 3, "Width": 1.5, "IsRed": 1},
        {"Length": 2, "Width": 1, "IsRed": 0},
        {"Length": 4, "Width": 1.5, "IsRed": 1},
        {"Length": 3, "Width": 1, "IsRed": 0},
        {"Length": 3.5, "Width": 0.5, "IsRed": 1},
        {"Length": 2, "Width": 0.5, "IsRed": 0},
        {"Length": 5.5, "Width": 1, "IsRed": 1},
        {"Length": 1, "Width": 1, "IsRed": 0},
        {"Length": 4.5, "Width": 1, "IsRed": 1},
    ]
    df = pd.DataFrame.from_records(data)

    y = np.array(df["IsRed"])
    X = df.copy()
    del X["IsRed"]

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    # define standard scaler
    scaler = StandardScaler()

    # transform data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ##########################################################

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense

    # Build a network model of sequential layers.
    model = Sequential()

    NUM_COLS = 2
    # Add 1st hidden layer. Note 1st hidden layer also receives data.
    # The input array must contain two feature columns and any number of rows.
    model.add(Dense(10, activation="sigmoid", input_shape=(NUM_COLS,)))

    # Add 2nd hidden layer.
    model.add(Dense(3, activation="sigmoid"))

    # Add output layer.
    model.add(Dense(1, activation="sigmoid"))

    # Compile the model.
    # Binary cross entropy is used to measure error cost for binary predictions.
    model.compile(loss="binary_crossentropy", metrics=["accuracy"])

    # Fit the model.
    # An epoch is one iteration for all samples through the network.
    # verbose can be set to 1 to show detailed output during training.
    model.fit(X_train_scaled, y_train, epochs=2000, verbose=1)

    # Evaluate the model.
    loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print("Test Accuracy: %.3f" % acc)

    # Make predictions.
    yhats = model.predict(X_test_scaled)
    print("Actual:")
    print(y_test)
    print("Predicted: ")
    print(yhats)
    predictions = []

    from sklearn.metrics import classification_report

    def showClassificationReport(y_test, yhats):
        # Convert continous predictions to
        # 0 or 1.
        for i in range(0, len(yhats)):
            if yhats[i] > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        print(classification_report(y_test, predictions))

    showClassificationReport(y_test, yhats)


def ex14():
    import torch
    import torch.optim as optim
    from sklearn.model_selection import train_test_split

    ### DATA           #######################################
    # Setup data.
    import pandas as pd
    import numpy as np

    # Load the flower feature data into a DataFrame.
    df = pd.DataFrame(columns=["Length", "Width", "IsRed"])
    data = [
        {"Length": 3, "Width": 1.5, "IsRed": 1},
        {"Length": 2, "Width": 1, "IsRed": 0},
        {"Length": 4, "Width": 1.5, "IsRed": 1},
        {"Length": 3, "Width": 1, "IsRed": 0},
        {"Length": 3.5, "Width": 0.5, "IsRed": 1},
        {"Length": 2, "Width": 0.5, "IsRed": 0},
        {"Length": 5.5, "Width": 1, "IsRed": 1},
        {"Length": 1, "Width": 1, "IsRed": 0},
        {"Length": 4.5, "Width": 1, "IsRed": 1},
    ]
    df = pd.DataFrame.from_records(data)

    y = np.array(df["IsRed"])
    X = df.copy()
    del X["IsRed"]
    X = X
    ##########################################################
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

    # Reshapes array.
    # unsqueeze() creates array of single dimensional arrays.
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    import torch
    import torch.nn as nn

    # Define the neural network architecture
    class BinaryClassificationNet(nn.Module):
        def __init__(self):
            super(BinaryClassificationNet, self).__init__()
            self.fc1 = nn.Linear(2, 8)
            self.fc2 = nn.Linear(8, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)  # Hidden layer.
            x = self.sigmoid(x)  # Activation function.
            x = self.fc2(x)  # Output layer.
            x = self.sigmoid(x)  # Activation function.
            return x

    # Instantiate the model
    model = BinaryClassificationNet()

    # Define the loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross entropy.
    # Use stochastic gradient descent to update weights & bias.
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    for epoch in range(2000):
        print("Epoch: " + str(epoch))
        # Forward pass
        output = model(X_train)
        loss = criterion(output, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    from sklearn.metrics import classification_report

    # Evaluate the model
    with torch.no_grad():
        predictions = []
        outputs = model(X_test)

        print("Actual:\n" + str(y_test.numpy()))
        print("Predictions:\n" + str(outputs.numpy()))

        yhats = outputs.numpy()
        for i in range(0, len(yhats)):
            if yhats[i] > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        print(classification_report(y_test, predictions))


def main():
    # ex4()  # FIX PLOT SIGMOID
    # ex10()
    # ex11()
    # ex13()
    ex14()


if __name__ == "__main__":
    main()
