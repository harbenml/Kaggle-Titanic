import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np
import config


def predict(test_data_path, model_type, model_path):
    df = pd.read_csv(test_data_path)
    test_idx = df["PassengerId"]
    predictions = None

    for FOLD in range(10):
        df = pd.read_csv(test_data_path)

        cols = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_columns.pkl"))
        df = df[cols]

        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 10

    sub = pd.DataFrame(
        np.column_stack((test_idx, predictions)), columns=["PassengerId", "Survived"]
    )
    sub["Survived"][sub["Survived"] >= 0.5] = 1
    sub["Survived"][sub["Survived"] < 0.5] = 0
    sub = sub.astype(int)
    return sub


if __name__ == "__main__":
    submission = predict(
        test_data_path=config.TEST_DATA, model_type="RF", model_path=config.MODEL_PATH,
    )

    submission.to_csv(f"{config.MODEL_PATH}rf_submission.csv", index=False)
