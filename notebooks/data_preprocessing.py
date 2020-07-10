# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python [conda env:ml] *
#     language: python
#     name: conda-env-ml-py
# ---

# # Kaggle Titanic Competition

# ## TODO
#
# - feature engineering
#      - family size = SibSp + Parch + 1
#      - Title
#      - Age = f(other features)

# ## Imports and helper functions

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


# -

def display_all(df):
    with pd.option_context("display.max_rows", 1500, "display.max_columns", 1000): 
        display(df)


# ## Load data

# load training data
train = pd.read_csv("../data/train.csv")

# load test data
test = pd.read_csv("../data/test.csv")

# ## Exploratory data analysis

train.shape

# We have:
# - 891 rows and
# - 12 columns

train.head()

# What kind of data types do we have in the data frame?

train.dtypes

display_all(train)

# Display the amount of NaN data per column

display_all( train.isnull().sum().sort_index()/len(train) )

train.describe()

# ## Data cleaning and feature engineering

# ### Remove labels from train set and merge train and test set for feature engineering

y = train.Survived
train.drop(['Survived'], axis=1, inplace=True)

data = pd.concat([train, test])
data.reset_index(inplace=True)

data.dtypes

# Convert categorical columns and do label encoding and inspect the data types again

# +
# label encoding
cols_to_encode = ['Sex', 'Cabin', 'Ticket', 'Embarked']
for col in cols_to_encode:
    data[col] = data[col].astype('category').cat.codes
    
data.dtypes
# -

data.drop('Name', axis=1)

# Replace NaNs with the median value

median_age = data["Age"].median()
data["Age"].fillna(median_age, inplace=True)

median_fare = data["Fare"].median()
data["Fare"].fillna(median_fare, inplace=True)

display_all( data.isna().sum().sort_index()/len(data) )

# ### Create training and test data

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = data.loc[:len(train)-1, features]
X_test = data.loc[len(train):, features]
print(y.shape)
print(X.shape)
print(X_test.shape)

train.tail()

X.tail()

# + [markdown] heading_collapsed=true
# ## Correlation analysis

# + hidden=true
corr = train.corr()
print(corr)

# + hidden=true
f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
# -

# # Modeling

# +

print(X.isna().any())
# -



SEED = 42
MODELS = {'SVC': make_pipeline(StandardScaler(), SVC(gamma="auto", random_state=SEED)),
          'RandomForest': RandomForestClassifier(n_jobs=-1,
                                                 n_estimators=1000,
                                                 max_features='sqrt',
                                                 min_samples_leaf=4,
                                                 oob_score=True,
                                                 random_state=RANDOM_STATE)}

score_means = []
for MODEL in MODELS:
    
    # training
    clf = MODELS[MODEL]
    clf.fit(X, y)
    train_performance = accuracy_score(y, clf.predict(X))
    print(f"Accuracy on training data: {train_performance :.5f}")
    
    # cross validation
    cv = StratifiedKFold(n_splits=10)
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=-1)
    score_means.append(scores.mean())
    print(f"Cross validation accuracy of {MODEL}: {scores.mean() :.5f} (+/- {scores.std()*2 :.5f})")
    
    # save model output on test data
    test["Survived"] = clf.predict(X_test)
    predictions = test[["PassengerId", "Survived"]]
    predictions.to_csv(f"model_{MODEL}.csv", index=False)





model_1 = pd.read_csv("model_RandomForest.csv")

model_1.head()

model_1.loc[6:10,:]



model_2 = pd.read_csv("model_SVC.csv")

model_2.head()

blend = round(0.49*model_1.Survived + 0.51*model_2.Survived)

blend.head()

sum(blend-model_2.Survived)



test["Survived"] = clf.predict(x_test)
submission = test[["PassengerId", "Survived"]]
submission.to_csv("submission.csv", index=False)

test

len(clf.predict(x_test))

# +
string = '  abc '

string.strip()
# -


