import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs/figures", exist_ok=True)

df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df.drop(
    "customerID",
    axis=1,
    inplace=True
)

print(df.head())


plt.figure(figsize=(6,4))

sns.countplot(
    x="Churn",
    data=df
)

plt.title("Customer Churn Distribution")

plt.savefig(
    "outputs/figures/churn_distribution.png"
)

plt.close()


plt.figure(figsize=(8,5))

sns.countplot(
    x="Contract",
    hue="Churn",
    data=df
)

plt.title("Churn by Contract Type")

plt.tight_layout()

plt.savefig(
    "outputs/figures/churn_by_contract.png"
)

plt.close()


plt.figure(figsize=(10,5))

sns.countplot(
    x="PaymentMethod",
    hue="Churn",
    data=df
)

plt.xticks(rotation=45)

plt.title(
    "Churn by Payment Method"
)

plt.tight_layout()

plt.savefig(
    "outputs/figures/churn_by_payment_method.png"
)

plt.close()


print("Starting Heatmap...")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Keep only numeric columns
numeric_df = df.select_dtypes(include=np.number)

print(numeric_df.columns)

corr = numeric_df.corr()

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "outputs/figures/correlation_heatmap.png"
)

plt.close()

print("Heatmap Done")

print("Creating Churn Correlation Plot...")

df_heat = pd.get_dummies(
    df,
    drop_first=True
)

corr = df_heat.corr()

churn_corr = corr["Churn_Yes"].sort_values()

plt.figure(figsize=(8,8))

churn_corr.plot(
    kind="barh"
)

plt.title(
    "Features Correlated with Churn"
)

plt.tight_layout()

plt.savefig(
    "outputs/figures/churn_correlation.png",
    dpi=300
)

plt.close()

print("Churn Correlation Plot Saved")


fig, axes = plt.subplots(
    1,
    2,
    figsize=(12,5)
)

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    ax=axes[0]
)

axes[0].set_title(
    "Tenure Distribution"
)

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    ax=axes[1]
)

axes[1].set_title(
    "Monthly Charges by Churn"
)

plt.tight_layout()

plt.savefig(
    "outputs/figures/tenure_charges.png"
)

plt.close()

print("PROGRAM FINISHED")
