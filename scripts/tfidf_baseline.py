"""Lab 3A starter: TF-IDF + LinearSVC baseline."""
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

DATA_PATH = Path("data/raw/bayan_feedback.csv")


def load_split(df: pd.DataFrame, split: str):
    subset = df[df["split"] == split]
    return subset["text"].tolist(), subset["topic"].tolist()


def main():
    df = pd.read_csv(DATA_PATH)

    X_train_text, y_train = load_split(df, "train")
    X_val_text, y_val = load_split(df, "validation")
    X_test_text, y_test = load_split(df, "test")

    # Fit the vectorizer ONLY on train text to avoid leaking val/test vocabulary.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    start = time.time()
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    val_pred = clf.predict(X_val)
    val_macro_f1 = f1_score(y_val, val_pred, average="macro")

    # Frozen test: evaluate once, do not use it to tune anything above.
    test_pred = clf.predict(X_test)
    test_macro_f1 = f1_score(y_test, test_pred, average="macro")

    print(f"Validation macro-F1: {val_macro_f1:.4f}")
    print(f"Frozen test macro-F1: {test_macro_f1:.4f}")
    print(f"Train time: {train_time:.2f}s")


if __name__ == "__main__":
    main()
