"""This module preprocesses the data for subsequent training and prediction steps.

The following procedure is applied:
1. Load the raw data.
2. Clean data by imputing missing values.
3. Create new features.
4. Encode features.
5. Export processed data and enocders.

"""
import pandas as pd
from sklearn import preprocessing
from typing import List, Tuple


# This dictionary is taken from
# https://medium.com/datadriveninvestor/start-with-kaggle-a-comprehensive-guide-to-solve-the-titanic-challenge-8ac5815b0473
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty",
}


class DataPreprocessing:
    def __init__(self, train_data_path: str = "", test_data_path: str = ""):
        self.train = pd.DataFrame()
        self.num_train_samples: int
        self.test = pd.DataFrame()
        self.full_data = pd.DataFrame()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.target_col = str

    def run_preprocessing(self):

        # load data
        self.get_data()
        self.target_col = "Survived"
        self.full_data = self.combine_train_test_set(
            self.train, self.test, self.target_col
        )

        # data cleaning and feature engineering
        self.create_new_features()
        self.clean_data()

        # label encoding of categorical features
        features, num_cols = self.get_features(self.full_data)
        self.full_data, _ = self.label_encode_features(
            self.full_data, features, num_cols
        )

        # split and export data
        self.train, self.test = self.split_train_test_set(
            self.full_data, self.target_col
        )
        self.export_data()

    def get_data(self):
        """Loads the data ,if paths are specified."""
        if self.train_data_path:
            self.train = pd.read_csv(self.train_data_path)
            self.num_train_samples = len(self.train)
            print("Train data loaded.")
        if self.test_data_path:
            self.test = pd.read_csv(self.test_data_path)
            print("Test data loaded.")

    def export_data(self):
        self.train.to_csv("data/processed/train.csv", index=False)
        self.test.to_csv("data/processed/test.csv", index=False)

    def create_new_features(self):
        # title and surname
        self.get_titles_and_surnames()
        # family size
        self.get_family_size()
        pass

    def clean_data(self):
        self.clean_missing_fare()
        self.clean_missing_age()
        self.clean_missing_embark()
        self.clean_missing_cabin()

    @staticmethod
    def combine_train_test_set(
        train: pd.DataFrame, test: pd.DataFrame, target_col
    ) -> pd.DataFrame:
        """Create dummy targets in test set and merge train and test set for feature engineering"""
        test.loc[:, target_col] = -1
        full_data = pd.concat([train, test]).reset_index(drop=True)
        return full_data

    @staticmethod
    def split_train_test_set(
        data: pd.DataFrame, target_col
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split full_data into training and test data"""
        train = data[data[target_col] != -1].reset_index(drop=True)
        test = data[data[target_col] == -1].reset_index(drop=True)
        test = test.drop(target_col, axis=1)
        return train, test

    def get_titles_and_surnames(self):
        """Get surname and title from name column."""
        surnames, titles = self._extract_features_from_names(self.full_data["Name"])
        # Replace Name column with the newly generated features
        self.full_data["Title"] = titles
        self.full_data["Surname"] = surnames
        self.full_data.drop("Name", axis=1, inplace=True)

    @staticmethod
    def _extract_features_from_names(names: List[str]) -> Tuple[List[str], List[str]]:
        surnames = [name.split(",")[0] for name in names]
        remainder = [name.split(",")[1].strip() for name in names]
        titles = pd.Series([x.split(".")[0] for x in remainder])
        # Reduce the number of titles.
        titles = titles.map(Title_Dictionary).to_list()
        return surnames, titles

    def get_family_size(self):
        """Create new feature Family Size."""
        self.full_data["Family_Size"] = (
            self.full_data["SibSp"] + self.full_data["Parch"] + 1
        )

    def clean_missing_fare(self):
        """Replace the missing value with the median Fare value of Pclass 3."""
        median_fare_pclass3 = self.full_data[self.full_data["Pclass"] == 3][
            "Fare"
        ].median()
        self.full_data["Fare"].fillna(median_fare_pclass3, inplace=True)

    def clean_missing_age(self):
        """Replace NaNs in Age column with the median value."""
        median_age = self.full_data["Age"].median()
        self.full_data["Age"].fillna(median_age, inplace=True)

    def clean_missing_embark(self):
        """Replace the NaNs with 'S'."""
        self.full_data["Embarked"].fillna("S", inplace=True)

    def clean_missing_cabin(self):
        """Take only the first letter of cabin values and replace the NaNs with 'NONE' string."""
        cabins = self.full_data.loc[~self.full_data["Cabin"].isnull(), "Cabin"]
        self.full_data.loc[~self.full_data["Cabin"].isnull(), "Cabin"] = cabins.map(
            lambda x: x[0]
        )
        self.full_data["Cabin"] = self.full_data["Cabin"].fillna("NONE").astype(str)

    @staticmethod
    def get_features(df: pd.DataFrame) -> (List[str], List[str]):
        # list of numerical columns
        num_cols = ["Age", "Fare", "Parch", "SibSp", "Family_Size"]
        # exclude the targets and PassengerId
        excluded_cols = ["Survived", "PassengerId"]
        # define features
        features = [f for f in df.columns if f not in excluded_cols]

        return features, num_cols

    @staticmethod
    def label_encode_features(
        df: pd.DataFrame, features: List[str], num_cols: List[str]
    ) -> pd.DataFrame:
        """For all categorical features, encode the categories to numerical values"""
        label_encoders = {}
        for col in features:
            if col not in num_cols:
                lbl = preprocessing.LabelEncoder()
                lbl.fit(df[col].values.tolist())
                df.loc[:, col] = lbl.transform(df[col].values.tolist())
                label_encoders[col] = lbl
        return df, label_encoders

    @staticmethod
    def fill_na_with_none(
        df: pd.DataFrame, features: List[str], num_cols: List[str]
    ) -> pd.DataFrame:
        """For each column, replace NaN values with NONE"""
        for col in features:
            if col not in num_cols:
                df.loc[:, col] = df[col].astype(str).fillna("NONE")
        return df


if __name__ == "__main__":
    obj = DataPreprocessing(
        train_data_path="data/raw/train.csv", test_data_path="data/raw/test.csv"
    )
    obj.run_preprocessing()

