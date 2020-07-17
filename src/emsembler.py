import os
from dispatcher import MODELS
import pandas as pd
import joblib
from config import (
    TEST_DATA_PATH,
    MODEL_PATH,
    NUM_FOLDS,
    FEATURE_COLS,
    TARGET_COL,
    SEED,
)


if __name__ == "__main__":
    predictions = None
    for i, MODEL in enumerate(MODELS):
        print(i, MODEL)
        model_preds = joblib.load(os.path.join(MODEL_PATH, f"{MODEL}_predictions.pkl"))

        if i == 0:
            predictions = model_preds
        else:
            predictions += model_preds

    predictions /= len(MODELS)

    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    predictions = predictions.astype(int)

    df = pd.read_csv(TEST_DATA_PATH)

    predictions = pd.concat(
        [df["PassengerId"], pd.Series(predictions, name="Survived")], axis=1
    )

    predictions.to_csv(f"{MODEL_PATH}ensemble_submission.csv", index=False)
