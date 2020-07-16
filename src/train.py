import joblib
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from typing import List
from config import (
    TRAIN_DATA_PATH,
    MODEL_PATH,
    NUM_FOLDS,
    FEATURE_COLS,
    TARGET_COL,
    SEED,
)

MODEL = "RF"


def run(fold: int) -> None:

    # read file
    df = pd.read_csv(TRAIN_DATA_PATH)

    # split train and val set
    df_train, df_val = get_train_val_set(df, fold)

    # get data for modeling
    X_train, y_train = df_train[FEATURE_COLS].values, df_train[TARGET_COL].values
    X_val, y_val = df_val[FEATURE_COLS].values, df_val[TARGET_COL].values

    model = RandomForestClassifier(
        n_jobs=-1,
        n_estimators=200,
        max_features="auto",
        min_samples_leaf=15,
        min_samples_split=10,
        oob_score=True,
        random_state=SEED,
    )

    # training
    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)[:, 1]
    auc = metrics.roc_auc_score(y_val, val_preds)
    print(f"Fold = {fold}, AUC = {auc}")

    joblib.dump(model, f"models/{MODEL}_{fold}.pkl")


def get_train_val_set(df: pd.DataFrame, fold: int) -> (pd.DataFrame, pd.DataFrame):
    """get training and validation data using folds"""
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_val = df[df.kfold == fold].reset_index(drop=True)
    return df_train, df_val


if __name__ == "__main__":
    for fold in range(NUM_FOLDS):
        run(fold)
