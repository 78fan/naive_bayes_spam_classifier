
import pandas as pd
import numpy as np
from bayes_classifier import NaiveBayesClassifier

df = pd.read_csv("spam.csv", encoding="latin1")
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
df = df.reindex(columns=["v2", "v1"])
df["v1"] = df["v1"].factorize()[0]
df = df.rename(columns={'v2': 'email', 'v1': 'spam'})
split_idx = int(len(df) * 0.9)

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

nbc = NaiveBayesClassifier(train_df)

train_df["predicted"] = train_df['email'].apply(nbc.classify)
print(f"Train accuracy: {(train_df["spam"] == train_df["predicted"]).mean()}")

test_df["predicted"] = test_df['email'].apply(nbc.classify)
print(f"Test accuracy: {(test_df["spam"] == test_df["predicted"]).mean()}")
