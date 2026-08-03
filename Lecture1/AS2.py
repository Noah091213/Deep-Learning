import pandas as pd

df = pd.read_csv("auto.csv")

df = df[df["mpg"] >= 16]

print(df[["weight", "acceleration"]].head(7))

df = df[df["horsepower"] != "?"]
df["horsepower"] = df["horsepower"].astype(int)

average = df.drop(columns=["name"]).mean()

print(average)

print(df.head())