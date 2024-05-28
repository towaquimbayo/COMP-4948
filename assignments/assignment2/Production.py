import pandas as pd
from keras.models import load_model
from pickle import load
import warnings

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
warnings.filterwarnings("ignore")


def prepare_data():
    # Keep only the relevant columns
    df = pd.read_csv("./test.csv")
    df = df[
        [
            "department",
            "previous_year_rating",
            "awards_won?",
            "avg_training_score",
        ]
    ]

    # Fill missing values in 'previous_year_rating'
    df["previous_year_rating"] = df["previous_year_rating"].fillna(0)

    # Label encode 'department'
    df["department"] = df["department"].astype("category").cat.codes

    return df


def main():
    try:
        # Prepare the data
        df = prepare_data()

        # Load the model
        model = load_model("model/best_model.keras")

        # Load the scaler
        with open("model/scaler.pkl", "rb") as file:
            scaler = load(file)

        # Scale the data
        x = df.values
        x = scaler.transform(x)

        # Make predictions
        predictions = model.predict(x)

        # Print the predictions
        print("Predictions:")
        for prediction in predictions:
            print(prediction)
    except FileNotFoundError as e:
        print("File not found:", e)
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return


if __name__ == "__main__":
    main()
