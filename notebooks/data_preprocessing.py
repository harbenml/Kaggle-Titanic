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
#     - ~~impute NaN in fare column of test set~~
# - feature engineering
#     - ~~family size = SibSp + Parch + 1~~
#     - ~~Title~~
#     - ~~~Age = f(other features)~~~
#

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
train = pd.read_csv("../data/raw/train.csv")

# load test data
test = pd.read_csv("../data/raw/test.csv")

# ## Exploratory data analysis

train.shape

# We have:
# - 891 rows and
# - 12 columns

train.head()

# What kind of data types do we have in the data frame?

train.dtypes

display_all(train)

display_all(test)

# Display the amount of NaN data per column in train set

train.isnull().sum().sort_index()/len(train)

# Display the amount of NaN data per column in test set

test.isnull().sum().sort_index()/len(test)

train.describe()

sns.pairplot(train)

# ## Data cleaning and feature engineering

# ### Remove labels from train set and merge train and test set for feature engineering

y = train.Survived
train.drop(['Survived'], axis=1, inplace=True)

data = pd.concat([train, test], sort=True).reset_index(drop=True)

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

# ### Reduce the number of titles

# This dictionary is taken from https://medium.com/datadriveninvestor/start-with-kaggle-a-comprehensive-guide-to-solve-the-titanic-challenge-8ac5815b0473

Title_Dictionary = {
    'Capt': 'Officer',
    'Col': 'Officer',
    'Major': 'Officer',
    'Jonkheer': 'Royalty',
    'Don': 'Royalty',
    'Sir' : 'Royalty',
    'Dr': 'Officer',
    'Rev': 'Officer',
    'the Countess':'Royalty',
    'Mme': 'Mrs',
    'Mlle': 'Miss',
    'Ms': 'Mrs',
    'Mr' : 'Mr',
    'Mrs' : 'Mrs',
    'Miss' : 'Miss',
    'Master' : 'Master',
    'Lady' : 'Royalty'
}

data['Title'] = data['Title'].map(Title_Dictionary)

# There is one sample in the test set with a title that is not covered in the dictionary.

data[data['Title'].isnull()==True]

# We will treat the title here as 'Royalty', because her full name is 'Oliva y Ocana, Dona. Fermina'.

data.loc[1305, 'Title'] = "Royalty"

# ### Create new feature Family Size

data['Family_Size'] = data['SibSp'] + data['Parch'] + 1

# ### Impute the missing Fare value

data[data['Fare'].isna()]

# Let's replace the missing value with the median Fare value of Pclass 3.

median_fare_pclass3 = data[data['Pclass']==3]['Fare'].median()
data["Fare"].fillna(median_fare_pclass3, inplace=True)
data.loc[1043]

# ### Replace NaNs in Age column with the median value

median_age = data["Age"].median()
data["Age"].fillna(median_age, inplace=True)

# ### Embarked column

data[data['Embarked'].isna()]

# What is the Emabrked value with the highest count?

data['Embarked'].value_counts()

# Replace the NaNs with 'S'

data["Embarked"].fillna('S', inplace=True)

# ### Cabin column

# Take only the first letter of cabin values and replace the NaNs with 'NONE' string

cabins = data.loc[~data['Cabin'].isnull(), 'Cabin']

data.loc[~data['Cabin'].isnull(), 'Cabin'] = cabins.map(lambda x: x[0])
data['Cabin'] = data['Cabin'].fillna("NONE").astype(str)

# ### Check, if everything is cleaned now

data.isna().sum().sort_index()/len(data)

display_all( data )

# ## Data export

# +
train = data.loc[:len(train)-1, :]
train = pd.concat([train, y], axis=1)
test = data.loc[len(train):, :]

train.to_csv('../data/processed/train.csv', index=False)
test.to_csv('../data/processed/test.csv', index=False)
# -

train.columns

# # Ressources

# - https://medium.com/datadriveninvestor/start-with-kaggle-a-comprehensive-guide-to-solve-the-titanic-challenge-8ac5815b0473
# - https://medium.com/i-like-big-data-and-i-cannot-lie/how-i-scored-in-the-top-9-of-kaggles-titanic-machine-learning-challenge-243b5f45c8e9
# - https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial

# + [markdown] heading_collapsed=true
# # Not used

# + [markdown] hidden=true
# ### Convert categorical columns (label encoding)

# + hidden=true
# label encoding
cols_to_encode = ['Sex', 'Cabin', 'Ticket', 'Embarked', 'Title', 'Surname']
for col in cols_to_encode:
    data[col] = data[col].astype('category').cat.codes + 1
    
data.dtypes

# + [markdown] hidden=true
# ### Make a regression model for age to impute missing values

# + hidden=true
age_model_input = data[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Family_Size']]

# + hidden=true
idx_age_isna = data['Age'].isnull()
sum(idx_age_isna)

# + hidden=true
age_model_train_input, age_targets = age_model_input[~idx_age_isna], data['Age'][~idx_age_isna]
age_model_inference_input = age_model_input[idx_age_isna]

# + hidden=true
len(age_model_train_input), len(age_targets)

# + hidden=true
len(age_model_inference_input)

# + hidden=true
len(data)-1046-263

# + hidden=true
SEED = 42
age_mdl = RandomForestRegressor(n_jobs=-1,
                                n_estimators=1000,
                                max_features='sqrt',
                                min_samples_leaf=10,
                                max_depth=5,
                                oob_score=True,
                                random_state=SEED)

# + hidden=true
age_mdl.fit(age_model_train_input, age_targets)

# + hidden=true
age_mdl.oob_score_

# + hidden=true
ages_to_impute = age_mdl.predict(age_model_inference_input)

# + hidden=true
# data.loc[idx_age_isna, 'Age'] = ages_to_impute

# + [markdown] hidden=true
# ## Feature transformation

# + [markdown] hidden=true
# The data consists of:
# - numerical features
# - nominal and ordinal categorical features

# + hidden=true
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']

# + [markdown] heading_collapsed=true
# # Modeling

# + [markdown] hidden=true
# ## Create training and test data

# + hidden=true
data.columns

# + hidden=true
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", 'Embarked', 'Title']

X = train.loc[:, features]
X_test = test.loc[:, features]
print(y.shape)
print(X.shape)
print(X_test.shape)

# + hidden=true
X.tail()

# + [markdown] hidden=true
# ## Correlation analysis

# + [markdown] hidden=true
# Before we do the training, we check, if there are some correlated features.

# + hidden=true
corr = X.corr()
print(corr)

# + hidden=true
f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

# + [markdown] hidden=true
# ## Training

# + hidden=true
print(X.isna().any())

# + hidden=true
SEED = 123
MODELS = {'SVC': make_pipeline(StandardScaler(), SVC(gamma="auto", random_state=SEED)),
          'RandomForest': RandomForestClassifier(n_jobs=-1,
                                                 n_estimators=2000,
                                                 max_features='sqrt',
                                                 min_samples_leaf=4,
                                                 oob_score=True,
                                                 random_state=SEED)}

# + hidden=true
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
