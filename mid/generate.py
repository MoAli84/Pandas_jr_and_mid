import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(123)

# Size of dataset
n = 1000

# Generate synthetic dataset
data_mid = {
    "employee_id": range(1, n+1),
    "full_name": np.random.choice(
        ["Ali Hassan", "Sara Mohamed", "Mona Adel", "Omar Fathy", "Nada Youssef",
         "Khaled Gamal", "Yara Ali", "Hossam Eldin", "Mai Sherif", None], size=n
    ),
    "department": np.random.choice(
        ["IT", "HR", "Finance", "Marketing", "Sales", "Operations", "Support"], size=n
    ),
    "city": np.random.choice(
        ["Cairo", "cairo", "Giza", "Alexandria", "alex", "Mansoura", "Tanta"], size=n
    ),
    "joining_date": pd.to_datetime(
        np.random.choice(pd.date_range("2015-01-01", "2023-12-31"), size=n)
    ),
    "age": np.random.choice(
        [22, 25, 28, 30, 32, 35, 38, 40, 45, None], size=n
    ),
    "monthly_salary": np.random.choice(
        [3000, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 15000, None],
        size=n,
        p=[0.05,0.1,0.1,0.1,0.1,0.15,0.1,0.1,0.05,0.05,0.1]  # some outliers & NaN
    ),
    "overtime_hours": np.random.randint(0, 60, size=n),
    "performance_score": np.random.choice(
        [1,2,3,4,5, None], size=n, p=[0.15,0.25,0.25,0.2,0.1,0.05]
    )
}

df_mid = pd.DataFrame(data_mid)

# Save to CSV
df_mid.to_csv("midlevel_dataset_1000.csv", index=False)

print("✅ Dataset saved as midlevel_dataset_1000.csv")
print(df_mid.head())
