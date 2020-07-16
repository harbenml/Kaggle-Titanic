import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np
from config import (
    TEST_DATA_PATH,
    MODEL_PATH,
    NUM_FOLDS,
    FEATURE_COLS,
    TARGET_COL,
    SEED,
)


def predict(model_type):
    df = pd.read_csv(TEST_DATA_PATH)
    test_idx = df["PassengerId"]
    predictions = None

    for FOLD in range(NUM_FOLDS):
        df = df[FEATURE_COLS]
        clf = joblib.load(os.path.join(MODEL_PATH, f"{model_type}_{FOLD}.pkl"))
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= NUM_FOLDS

    sub = pd.DataFrame(
        np.column_stack((test_idx, predictions)), columns=["PassengerId", "Survived"]
    )
    sub["Survived"][sub["Survived"] >= 0.5] = 1
    sub["Survived"][sub["Survived"] < 0.5] = 0
    sub = sub.astype(int)
    return sub


if __name__ == "__main__":
    submission = predict(model_type="RF")
    submission.to_csv(f"{MODEL_PATH}rf_submission.csv", index=False)
