def ex1():
    import numpy as np
    import pandas as pd
    import itertools
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Read the data
    df = pd.read_csv("/Users/elber/Documents/COMP 4948 - Datasets/iris_v3.csv")

    # Get shape and head
    print(df.shape)
    print(df.head())
    pd.set_option("display.max_colwidth", None)
    print(df.head(1))

    X = df.copy()
    del X["iris_type"]
    y = df["iris_type"]
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize a PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(C=5)
    pac.fit(X_train, y_train)

    # - Predict on the test set and calculate accuracy
    y_pred = pac.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {round(score*100,2)}%")

    # Creating classification report
    print(classification_report(y_test, y_pred))


def ex2():
    import numpy as np
    import pandas as pd
    import itertools
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Read the data
    df = pd.read_csv("/Users/elber/Documents/COMP 4948 - Datasets/FakeNews.csv")

    # Get shape and head
    print(df.shape)
    print(df.head())
    pd.set_option("display.max_colwidth", None)
    print(df.head(1))

    # Fake Articles
    print("Number of FAKE articles: " + str(np.sum(df["label"] == "FAKE")))

    # Real Articles
    print("Number of REAL articles: " + str(np.sum(df["label"] == "REAL")))

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=7
    )

    # Initialize a TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

    # Fit and transform train set, transform test set
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)

    # Initialize a PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier()
    pac.fit(tfidf_train, y_train)

    # - Predict on the test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {round(score*100,2)}%")

    # Build confusion matrix
    confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])

    # Creating classification report
    print(classification_report(y_test, y_pred))

    # test with 2 news items
    data = np.array(
        ["Towa Quimbayo is going to space.", "Towa Quimbayo works at BCIT."]
    )
    series = pd.Series(data)
    tfidf_test = tfidf_vectorizer.transform(series)
    y_pred = pac.predict(tfidf_test)
    print(y_pred)


def main():
    # ex1()
    ex2()


if __name__ == "__main__":
    main()
