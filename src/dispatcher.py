from train_sklearn import Sklearn
from config import VERBOSE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


rf_params = {
    "n_jobs": -1,
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "verbose": VERBOSE,
}
gb_params = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_leaf": 3,
    "verbose": VERBOSE,
}
svc_params = {
    "kernel": "linear",
    "C": 0.025,
    "probability": True,
    "verbose": VERBOSE,
}


MODELS = {
    "randomforest": Sklearn(clf=RandomForestClassifier, params=rf_params),
    "gradientboost": Sklearn(clf=GradientBoostingClassifier, params=gb_params),
    "svc": Sklearn(clf=SVC, params=svc_params),
}
