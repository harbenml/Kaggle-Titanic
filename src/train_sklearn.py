import pandas as pd
from config import SEED


class Sklearn(object):
    def __init__(self, clf, params=None):
        params["random_state"] = SEED
        self.clf = clf(**params)

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        self.clf.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.clf.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.clf.predict_proba(X)

