import joblib
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from typing import List

SEED = 23


def run(fold: int) -> None:

    df = pd.read_csv("data/processed/train_folds.csv")

    features, num_cols = get_features(df)
    df = fill_na_with_none(df, features, num_cols)
    df = label_encode_features(df, features, num_cols)
    df_train, df_val = get_train_val_set(df, fold)
    X_train, y_train = df_train[features].values, df_train["Survived"].values
    X_val, y_val = df_val[features].values, df_val["Survived"].values

    model = RandomForestClassifier(
        n_jobs=-1,
        n_estimators=1000,
        max_features="sqrt",
        min_samples_leaf=4,
        oob_score=True,
        random_state=SEED,
    )

    # training
    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)[:, 1]
    auc = metrics.roc_auc_score(y_val, val_preds)
    print(f"Fold = {fold}, AUC = {auc}")

    # train_performance = accuracy_score(y, clf.predict(X))
    # print(f"Accuracy on training data: {train_performance :.5f}")

    # # cross validation
    # cv = StratifiedKFold(n_splits=10)
    # scores = cross_val_score(clf, X, y, cv=cv, n_jobs=-1)
    # score_means.append(scores.mean())
    # print(
    #     f"Cross validation accuracy of {MODEL}: {scores.mean() :.5f} (+/- {scores.std()*2 :.5f})"
    # )

    # print(features)


def get_features(df: pd.DataFrame) -> (List[str], List[str]):
    # list of numerical columns
    num_cols = ["Age", "Fare"]
    # exclude the targets, kfolds and PassengerId
    excluded_cols = ["Survived", "kfold", "PassengerId"]
    # define features
    features = [f for f in df.columns if f not in excluded_cols]

    return features, num_cols


def fill_na_with_none(
    df: pd.DataFrame, features: List[str], num_cols: List[str]
) -> pd.DataFrame:
    """For each column, replace NaN values with NONE"""
    for col in features:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")
    return df


def label_encode_features(
    df: pd.DataFrame, features: List[str], num_cols: List[str]
) -> pd.DataFrame:
    """For all categorical features, encode the categories to numerical values"""
    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:, col] = lbl.transform(df[col])

    return df


def get_train_val_set(df: pd.DataFrame, fold: int) -> (pd.DataFrame, pd.DataFrame):
    """get training and validation data using folds"""
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_val = df[df.kfold == fold].reset_index(drop=True)
    return df_train, df_val


if __name__ == "__main__":
    for fold in range(5):
        run(fold)
