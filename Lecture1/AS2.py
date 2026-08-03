import pandas as pd
import os

# Read the CSV file using the path relative to this Python file
df = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/auto.csv")

# Remove all rows where 'mpg' is lower than 16
df = df[df["mpg"] >= 16]

# Display the first 7 rows of the 'weight' and 'acceleration' columns
print(df[["weight", "acceleration"]].head(7))

# Remove rows where 'horsepower' contains '?'
df = df[df["horsepower"] != "?"]

# Converts the 'horsepower' column from string to integer
df["horsepower"] = df["horsepower"].astype(int)

# Calculates the average of all columns except 'name'
average = df.drop(columns=["name"]).mean()

# Prints averages
print(average)

# Displays the first 5 rows of the filtered Data
print(df.head())
