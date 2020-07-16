TRAIN_DATA_PATH = "data/processed/train_folds.csv"
TEST_DATA_PATH = "data/processed/test.csv"
MODEL_PATH = "models/"
NUM_FOLDS = 5
SEED = 23
FEATURE_COLS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
    "Title",
    "Surname",
    "Family_Size",
]
TARGET_COL = "Survived"
