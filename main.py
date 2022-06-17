import pandas as pd
from sklearn.model_selection import train_test_split

# Hyperparameters for the model
TRAIN_PERCENT = .9
VAL_PERCENT = .3

def dfsplit(df):
    train_df, test_df = train_test_split(df, train_size=TRAIN_PERCENT)
    train_df, val_df = train_test_split(train_df, test_size=VAL_PERCENT)
    return train_df, val_df, test_df

def preprocess(file_path, categories_filter):
    arxiv_df = pd.read_json(file_path, lines=True)
    # Drop all columns that are not abstract and title
    arxiv_df = arxiv_df[["abstract", "title", "categories"]].drop_duplicates(subset=["abstract"])

    # Collect all distinct categories then drop the column
    pat = '|'.join(r"\b{}\b".format(x) for x in categories_filter)
    arxiv_df = arxiv_df[arxiv_df["categories"].str.contains(pat)]
    arxiv_df = arxiv_df[["abstract", "title"]]

    return arxiv_df

def savedf(df):
    # Perform data splitting
    train_df, val_df, test_df = dfsplit(df)
    assert(df.shape[0] == train_df.shape[0] + val_df.shape[0] + test_df.shape[0])

    # Save datasets to csv files
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)


if __name__ == "__main__":
    categories = ["cs."]
    arxiv_df = preprocess("data/arxiv-metadata-oai-snapshot.json", categories)
    savedf(arxiv_df);
