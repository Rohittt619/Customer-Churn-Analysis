import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# Churn Distribution
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Distribution")
plt.savefig("images/churn_distribution.png")
plt.show()

