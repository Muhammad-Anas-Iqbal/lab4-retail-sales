import pandas as pd
import numpy as np
from scipy.stats import zscore

data = {
    "Day": [
        "Day 1", "Day 2", "Day 3", "Day 4", "Day 5",
        "Day 6", "Day 7", "Day 8", "Day 9", "Day 10"
    ],

    "Revenue": [
        120, 135, np.nan, 128, 142,
        130, 500, np.nan, 138, 125
    ]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)


# (a) Missing-value handling
print("\nMissing Values:")
print(df.isnull().sum())

# Calculate median Revenue
median_revenue = df["Revenue"].median()

print("\nMedian Revenue:")
print(median_revenue)

# Fill missing values using median
df["Revenue"] = df["Revenue"].fillna(median_revenue)

print("\nDataset After Filling Missing Values:")
print(df)


# (b) Min-Max Normalization and Z-score Standardization

# Min-Max normalization
min_revenue = df["Revenue"].min()
max_revenue = df["Revenue"].max()

df["Revenue_MinMax"] = (
    (df["Revenue"] - min_revenue) /
    (max_revenue - min_revenue)
)

# Z-score standardization
df["Revenue_Zscore"] = zscore(df["Revenue"])


# (c) IQR-Based Outlier Detection

Q1 = df["Revenue"].quantile(0.25)
Q3 = df["Revenue"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nIQR Information:")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# Detect outliers
df["Outlier"] = (
    (df["Revenue"] < lower_bound) |
    (df["Revenue"] > upper_bound)
)

# Print flagged days
outliers = df[df["Outlier"]]

print("\nDetected Outlier Day(s):")

if outliers.empty:
    print("No outliers detected.")
else:
    print(outliers[["Day", "Revenue"]])


# --------------------------------------------------
# (d) Final Table
# --------------------------------------------------

print("\nFinal Table:")
print(
    df[
        [
            "Day",
            "Revenue",
            "Revenue_MinMax",
            "Revenue_Zscore",
            "Outlier"
        ]
    ]
)