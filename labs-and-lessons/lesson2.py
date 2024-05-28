def ex4():
    import sklearn.datasets as datasets
    import matplotlib.pyplot as plt

    # check for the sklearn version, it has to be 0.21
    import sklearn

    print(sklearn.__version__)
    breast_cancer = datasets.load_breast_cancer()

    import pandas as pd

    dfX = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    dfy = breast_cancer.target
    print(dfX.head())
    print(dfy)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        dfX, dfy, test_size=0.20, random_state=0
    )

    from sklearn.tree import DecisionTreeClassifier

    classifier = DecisionTreeClassifier(max_depth=2)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    from sklearn.metrics import confusion_matrix

    def showAccuracyScores(y_test, y_pred):
        print("\nModel Evaluation")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print("")
        tn = cm[0][0]
        fp = cm[0][1]
        tp = cm[1][1]
        fn = cm[1][0]
        accuracy = (tp + tn) / (tn + fp + tp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print("Accuracy: " + str(accuracy))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))

    showAccuracyScores(y_test, y_pred)

    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(20, 10))

    plot_tree(classifier.fit(X_train, y_train), max_depth=4, fontsize=4)
    a = plot_tree(
        classifier,
        feature_names=list(breast_cancer.feature_names),
        filled=True,
        rounded=True,
        fontsize=14,
    )
    plt.show()


def main():
    ex4()


if __name__ == "__main__":
    main()
