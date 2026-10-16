
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset and trained model paths
DATA_PATH = BASE_DIR / "ml" / "reviews.csv"
MODEL_PATH = BASE_DIR / "ml" / "sentiment_model.joblib"


# Sample dataset used when reviews.csv is missing or empty
SAMPLE_REVIEWS = [
    # Positive reviews
    ("I love this product. It works perfectly.", "positive"),
    ("Excellent quality and amazing performance.", "positive"),
    ("The experience was wonderful and satisfying.", "positive"),
    ("Very happy with my purchase.", "positive"),
    ("The service was excellent and fast.", "positive"),
    ("Absolutely fantastic product.", "positive"),
    ("Great value for money.", "positive"),
    ("I would definitely recommend this.", "positive"),
    ("The quality is superb.", "positive"),
    ("I am very pleased with the results.", "positive"),
    ("Beautiful design and excellent features.", "positive"),
    ("The delivery was quick and reliable.", "positive"),
    ("This is the best product I have purchased.", "positive"),
    ("Works smoothly and saves time.", "positive"),
    ("Outstanding customer support.", "positive"),

    # Negative reviews
    ("I hate this product. It is terrible.", "negative"),
    ("Very poor quality and disappointing performance.", "negative"),
    ("The experience was awful.", "negative"),
    ("I am unhappy with my purchase.", "negative"),
    ("The service was slow and frustrating.", "negative"),
    ("Completely useless product.", "negative"),
    ("Not worth the money.", "negative"),
    ("I would not recommend this product.", "negative"),
    ("The quality is horrible.", "negative"),
    ("I regret buying this item.", "negative"),
    ("The product stopped working after one day.", "negative"),
    ("Very bad customer support.", "negative"),
    ("The delivery was late and the package was damaged.", "negative"),
    ("This product does not work properly.", "negative"),
    ("A disappointing and frustrating experience.", "negative"),

    # Neutral reviews
    ("The product arrived yesterday.", "neutral"),
    ("The package contains three items.", "neutral"),
    ("The product is available in two colors.", "neutral"),
    ("I received the order this morning.", "neutral"),
    ("The device has a screen and two buttons.", "neutral"),
    ("The shop opens at nine in the morning.", "neutral"),
    ("The item weighs approximately two kilograms.", "neutral"),
    ("The product comes with a user manual.", "neutral"),
    ("The order number is printed on the receipt.", "neutral"),
    ("The package was delivered to my address.", "neutral"),
    ("The product has several different settings.", "neutral"),
    ("The meeting is scheduled for tomorrow.", "neutral"),
    ("The box contains a charging cable.", "neutral"),
    ("The website lists the available sizes.", "neutral"),
    ("I purchased this item last week.", "neutral"),
]


def load_training_data():
    """Load a valid CSV dataset or use the built-in sample dataset."""

    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        try:
            data = pd.read_csv(DATA_PATH)
        except pd.errors.EmptyDataError:
            print("Warning: reviews.csv is empty. Using sample data.")
            data = pd.DataFrame(
                SAMPLE_REVIEWS,
                columns=["text", "sentiment"],
            )
    else:
        print("Warning: reviews.csv is missing or empty. Using sample data.")
        data = pd.DataFrame(
            SAMPLE_REVIEWS,
            columns=["text", "sentiment"],
        )

    # Validate the required columns
    required_columns = {"text", "sentiment"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "reviews.csv must contain these columns: text, sentiment"
        )

    if data.empty:
        print("Warning: No training rows found. Using sample data.")
        data = pd.DataFrame(
            SAMPLE_REVIEWS,
            columns=["text", "sentiment"],
        )

    # Clean the dataset
    data = data.dropna(subset=["text", "sentiment"]).copy()

    data["text"] = data["text"].astype(str).str.strip()
    data["sentiment"] = (
        data["sentiment"].astype(str).str.lower().str.strip()
    )

    data = data[data["text"] != ""]

    # Validate sentiment labels
    allowed_sentiments = {"positive", "negative", "neutral"}

    invalid_labels = set(data["sentiment"]) - allowed_sentiments

    if invalid_labels:
        raise ValueError(
            f"Invalid sentiment labels: {invalid_labels}. "
            "Use positive, negative, or neutral."
        )

    if data.empty:
        raise ValueError("No valid training data is available.")

    if data["sentiment"].nunique() < 3:
        raise ValueError(
            "Training data must contain positive, negative, "
            "and neutral examples."
        )

    return data


def train_model():
    """Train and save the sentiment analysis pipeline."""

    data = load_training_data()

    # Create the NLP machine learning pipeline
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # Train the model
    model.fit(data["text"], data["sentiment"])

    # Save the trained model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\n" + "=" * 45)
    print("SENTIMENT ANALYSIS MODEL TRAINED SUCCESSFULLY")
    print("=" * 45)
    print(f"Training samples : {len(data)}")
    print(f"Sentiment classes: {list(model.classes_)}")
    print(f"Model saved to   : {MODEL_PATH}")

    # Test a few example sentences
    test_sentences = [
        "I absolutely love this product!",
        "This is a terrible experience.",
        "The package arrived this morning.",
    ]

    print("\nSample predictions:")

    for sentence in test_sentences:
        prediction = model.predict([sentence])[0]
        print(f"Text       : {sentence}")
        print(f"Sentiment  : {prediction}")
        print("-" * 45)


if __name__ == "__main__":
    train_model()

