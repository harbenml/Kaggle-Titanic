import joblib
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from typing import List
import config

SEED = config.SEED
MODEL = "RF"


def run(fold: int) -> None:

    # read file
    df = pd.read_csv("data/processed/train_folds.csv")

    # split train and val set
    df_train, df_val = get_train_val_set(df, fold)

    # get features
    excluded_cols = ["Survived", "PassengerId", "kfold"]
    features = [f for f in df.columns if f not in excluded_cols]

    # get data for modeling
    X_train, y_train = df_train[features].values, df_train["Survived"].values
    X_val, y_val = df_val[features].values, df_val["Survived"].values

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
    joblib.dump(features, f"models/{MODEL}_{fold}_columns.pkl")


def get_train_val_set(df: pd.DataFrame, fold: int) -> (pd.DataFrame, pd.DataFrame):
    """get training and validation data using folds"""
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_val = df[df.kfold == fold].reset_index(drop=True)
    return df_train, df_val


if __name__ == "__main__":
    for fold in range(10):
        run(fold)
