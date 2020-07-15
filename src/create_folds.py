import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":

    # load training data
    df = pd.read_csv("data/processed/train.csv")

    # create a new column kfold
    df["kfold"] = -1

    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # get the target values
    y = df["Survived"].values

    # do stratified kfold in order to have equally distributed folds
    kf = model_selection.StratifiedKFold(n_splits=10)

    # get values for the kfolds
    for fold, (train_idx, val_idx) in enumerate(kf.split(df, y)):
        df.loc[val_idx, "kfold"] = fold

    # save df with new column kfold
    df.to_csv("data/processed/train_folds.csv", index=False)

