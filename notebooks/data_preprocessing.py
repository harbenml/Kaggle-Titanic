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
# - data cleaning
#     - impute NaN in fare column of test set
# - feature engineering
#     - family size = SibSp + Parch + 1
#     - Title
#     - Age = f(other features)

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
from sklearn.ensemble import RandomForestRegressor


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

# Display the amount of NaN data per column in train set

display_all( train.isnull().sum().sort_index()/len(train) )

# Display the amount of NaN data per column in test set

display_all( test.isnull().sum().sort_index()/len(test) )

train.describe()

sns.pairplot(train)

# ## Data cleaning and feature engineering

# ### Remove labels from train set and merge train and test set for feature engineering

y = train.Survived
train.drop(['Survived'], axis=1, inplace=True)

data = pd.concat([train, test])
data.reset_index(inplace=True)

data.dtypes


# ### Get surname and title from name column

def extract_features_from_names(names):
    surnames = [name.split(',')[0] for name in names]
    remainder = [name.split(',')[1].strip() for name in names]
    titles = [x.split('.')[0] for x in remainder]
    return surnames, titles


surnames, titles = extract_features_from_names(data['Name'])

# Unique titles in names:

print(set(titles))

# Replace *Name* columns with the newly generated features

data['Title'] = titles
data['Surname'] = surnames

data.drop('Name', axis=1, inplace=True)

# ### Create new feature Family Size

data['Family_Size'] = data['SibSp'] + data['Parch'] + 1

# ### Convert categorical columns (label encoding)

# +
# label encoding
cols_to_encode = ['Sex', 'Cabin', 'Ticket', 'Embarked', 'Title', 'Surname']
for col in cols_to_encode:
    data[col] = data[col].astype('category').cat.codes + 1
    
data.dtypes
# -

# ### Impute the missing Fare value

data[data['Fare'].isna()]

# Let's replace the missing value with the median Fare value of Pclass 3.

median_fare_pclass3 = data[data['Pclass']==3]['Fare'].median()
data["Fare"].fillna(median_fare_pclass3, inplace=True)
data.loc[1043]



# ### Make a regression model for age to impute missing values

age_model_input = data[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Family_Size']]

idx_age_isna = data['Age'].isnull()
sum(idx_age_isna)

age_model_train_input, age_targets = age_model_input[~idx_age_isna], data['Age'][~idx_age_isna]
age_model_inference_input = age_model_input[idx_age_isna]

len(age_model_train_input), len(age_targets)

len(age_model_inference_input)

len(data)-1046-263

SEED = 42
age_mdl = RandomForestRegressor(n_jobs=-1,
                                n_estimators=1000,
                                max_features='sqrt',
                                min_samples_leaf=10,
                                max_depth=5,
                                oob_score=True,
                                random_state=SEED)

age_mdl.fit(age_model_train_input, age_targets)

age_mdl.oob_score_

ages_to_impute = age_mdl.predict(age_model_inference_input)

data.loc[idx_age_isna, 'Age'] = ages_to_impute

display_all( data )

# ### Check, if everything is cleaned now

display_all( data.isna().sum().sort_index()/len(data) )

# ## Create training and test data

data.columns

# +
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", 'Embarked', 'Title']

X = data.loc[:len(train)-1, features]
X_test = data.loc[len(train):, features]
print(y.shape)
print(X.shape)
print(X_test.shape)
# -

X.tail()

# ## Correlation analysis

corr = X.corr()
print(corr)

f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

# # Modeling

print(X.isna().any())

SEED = 123
MODELS = {'SVC': make_pipeline(StandardScaler(), SVC(gamma="auto", random_state=SEED)),
          'RandomForest': RandomForestClassifier(n_jobs=-1,
                                                 n_estimators=2000,
                                                 max_features='sqrt',
                                                 min_samples_leaf=4,
                                                 oob_score=True,
                                                 random_state=SEED)}

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




