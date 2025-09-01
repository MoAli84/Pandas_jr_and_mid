import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load & Explore the Data
# -------------------------------
df = pd.read_csv("data.csv")

print(df.head())
print(df.info())
print(df.describe())
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# -------------------------------
# 2. Data Quality Check
# -------------------------------
# Missing values
print("Missing values:\n", df.isna().sum())

df["name"].fillna("Unknown", inplace=True)
df["age"].fillna(df["age"].median(), inplace=True)
df["salary"].fillna(df["salary"].median(), inplace=True)

# Duplicates
print("Duplicate rows:", df.duplicated().sum())

# Quick frequency counts for key columns
for col in ["name", "age", "city", "salary", "date"]:
    print(f"\nValue counts for {col}:\n", df[col].value_counts().head(10))

# Fix invalid dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Check year range validity
print("Dates in 2020–2023 range:\n",
      df["date"].dt.year.between(2020, 2023).value_counts())

# -------------------------------
# 3. Basic Insights
# -------------------------------
print("Unique employees (ID):", df["id"].nunique())
print("Average Salary:", df["salary"].mean())
print("Minimum Salary:", df["salary"].min())
print("Maximum Salary:", df["salary"].max())
print("Average Age:", df["age"].mean())

print("\nEmployees per city:\n", df["city"].value_counts())
print("\nMost common names:\n", df["name"].value_counts().head(5))

# -------------------------------
# 4. Time-based Analysis
# -------------------------------
df["year"] = df["date"].dt.year
print("\nRecords per year:\n", df["year"].value_counts())
print("\nAverage salary per year:\n", df.groupby("year")["salary"].mean().round(2))

sns.countplot(data=df, x="year")
plt.title("Employee Records Distribution per Year")
plt.show()

# -------------------------------
# 5. Segmentation
# -------------------------------
print("\nCities by highest average salary:\n",
      df.groupby("city")["salary"].mean().sort_values(ascending=False))

print("\nCity with youngest workforce:\n",
      df.groupby("city")["age"].mean().sort_values().head(1))

print("\nTop 5 earners:\n",
      df.nlargest(5, "salary")[["name", "city", "salary"]])

# -------------------------------
# 6. Data Cleaning
# -------------------------------
df["name"] = df["name"].str.strip()        # remove extra spaces
df.drop_duplicates(inplace=True)           # remove duplicates

# Save cleaned dataset
df.to_csv("cleaned_data.csv", index=False)

print("\n✅ Data cleaning complete. Cleaned file saved as 'cleaned_data.csv'")
