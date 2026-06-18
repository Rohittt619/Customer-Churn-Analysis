# 📉 Customer Churn Analysis

> Predicting which customers are likely to leave — using Exploratory Data Analysis, visualizations, and Machine Learning in Python.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Problem Statement

Customer churn is when a customer stops doing business with a company. Acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds a system to:

- Identify the **key drivers** that cause customers to churn
- Predict which customers are **at risk of churning** before they leave
- Provide **actionable business insights** to reduce churn rate

**Dataset:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
7,043 customers · 21 features · Binary classification (Churn: Yes / No)

---

## 🗂️ Project Structure

```
customer-churn-analysis/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # raw dataset
│   └── cleaned_churn.csv                        # after preprocessing
│
├── src/
│   ├── preprocess.py                            # data cleaning functions
│   ├── eda.py                                   # EDA helper functions
│   └── model.py                                 # model training pipeline
│
├── outputs/
│   ├── figures/                                 # saved charts (PNG)
│   └── model_results.csv                        # model comparison table
│
├── requirements.txt
└── README.md
```

---

## 🔍 Dataset Overview

| Feature | Description |
|---|---|
| `customerID` | Unique customer identifier |
| `tenure` | Number of months as a customer |
| `MonthlyCharges` | Monthly billing amount (₹) |
| `TotalCharges` | Total amount charged |
| `Contract` | Month-to-month / One year / Two year |
| `InternetService` | DSL / Fiber optic / No |
| `PaymentMethod` | Electronic check / Mailed check / etc. |
| `Churn` | **Target variable** — Yes / No |

**Key stats:**
- Churn rate: ~26.5% (class imbalance present)
- Average tenure of churned customers: 18 months vs 38 months for retained
- Fiber optic users churn at nearly 2× the rate of DSL users

---

## 🧹 Data Preprocessing

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# TotalCharges had whitespace strings instead of NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop customerID — not a feature
df.drop('customerID', axis=1, inplace=True)

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encode categorical columns
df = pd.get_dummies(df, drop_first=True)
```

**Preprocessing steps:**
- Fixed `TotalCharges` column — had 11 blank strings stored as whitespace
- Removed `customerID` (non-predictive identifier)
- Encoded all categorical variables using one-hot encoding
- Scaled numerical features using `StandardScaler` for distance-based models
- Verified no remaining null values: `df.isnull().sum().sum() == 0`

---

## 📊 Exploratory Data Analysis

### Churn distribution

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(6, 4))
df['Churn'].value_counts().plot(kind='bar', color=['#185FA5', '#E24B4A'], ax=ax)
ax.set_title('Customer Churn Distribution')
ax.set_xlabel('Churn (0 = No, 1 = Yes)')
ax.set_ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/figures/churn_distribution.png', dpi=150)
plt.show()
```

### Key findings from EDA

| Insight | Detail |
|---|---|
| **Contract type is the strongest churn driver** | Month-to-month customers churn at 42% vs 11% for 1-year contracts |
| **Tenure is inversely correlated with churn** | New customers (0–12 months) churn at 3× the rate of long-term ones |
| **Electronic check users churn most** | 45% churn rate vs ~17% for auto-payment methods |
| **Fiber optic has a churn problem** | 41% churn rate — higher monthly charges may drive this |
| **No tech support = higher churn** | Customers without tech support churn at 41% vs 15% |

```python
# Churn rate by contract type
contract_churn = df.groupby('Contract')['Churn'].mean().reset_index()
contract_churn.columns = ['Contract', 'Churn Rate']
contract_churn['Churn Rate'] = (contract_churn['Churn Rate'] * 100).round(1)

sns.barplot(data=contract_churn, x='Contract', y='Churn Rate', palette='Blues_d')
plt.title('Churn Rate by Contract Type')
plt.ylabel('Churn Rate (%)')
plt.tight_layout()
plt.savefig('outputs/figures/churn_by_contract.png', dpi=150)
plt.show()
```

### Correlation heatmap

```python
plt.figure(figsize=(12, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='Blues', linewidths=0.5,
            cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('outputs/figures/correlation_heatmap.png', dpi=150)
plt.show()
```

### Tenure distribution by churn status

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
df[df['Churn']==0]['tenure'].plot(kind='hist', bins=30, ax=axes[0],
    color='#185FA5', alpha=0.7, label='Retained')
df[df['Churn']==1]['tenure'].plot(kind='hist', bins=30, ax=axes[0],
    color='#E24B4A', alpha=0.7, label='Churned')
axes[0].set_title('Tenure Distribution by Churn')
axes[0].set_xlabel('Tenure (months)')
axes[0].legend()

# Box plot
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=axes[1],
            palette={0:'#185FA5', 1:'#E24B4A'})
axes[1].set_title('Monthly Charges by Churn Status')
axes[1].set_xticklabels(['Retained', 'Churned'])

plt.tight_layout()
plt.savefig('outputs/figures/tenure_charges.png', dpi=150)
plt.show()
```

---

## 🤖 Machine Learning Models

Three models were trained and compared. Class imbalance was handled using `class_weight='balanced'`.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Train-test split
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Models
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1]):.4f}")
```

---

## 📈 Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 79.2% | 0.63 | 0.76 | 0.69 | 0.847 |
| Random Forest | 81.4% | 0.67 | 0.72 | 0.69 | 0.862 |
| **Gradient Boosting** | **82.1%** | **0.69** | **0.74** | **0.71** | **0.871** |

> **Best model: Gradient Boosting** — highest ROC-AUC (0.871) and F1-score (0.71).

> **Why F1-score matters here:** In churn prediction, missing a churner (false negative) is costly. F1-score balances precision and recall — better than accuracy alone on imbalanced data.

### Confusion matrix — Gradient Boosting

```python
from sklearn.metrics import ConfusionMatrixDisplay

best_model = models['Gradient Boosting']
y_pred_best = best_model.predict(X_test_scaled)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best,
    display_labels=['Retained', 'Churned'],
    cmap='Blues', ax=ax
)
ax.set_title('Confusion Matrix — Gradient Boosting')
plt.tight_layout()
plt.savefig('outputs/figures/confusion_matrix.png', dpi=150)
plt.show()
```

### Feature importance

```python
rf_model = models['Random Forest']
feature_importance = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
feature_importance.plot(kind='barh', color='#185FA5')
plt.title('Top 10 Features — Random Forest')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/figures/feature_importance.png', dpi=150)
plt.show()
```

**Top 5 features driving churn:**
1. `tenure` — longer tenure = much lower churn risk
2. `MonthlyCharges` — higher bills = higher churn
3. `Contract_Two year` — long contracts strongly reduce churn
4. `InternetService_Fiber optic` — fiber customers churn more
5. `PaymentMethod_Electronic check` — this payment method flags at-risk customers

---

## 💡 Business Recommendations

Based on EDA and model findings, here are three actionable strategies:

| Strategy | Target Segment | Expected Impact |
|---|---|---|
| **Offer long-term contract incentives** | Month-to-month customers in months 1–6 | Could reduce churn rate from 42% → ~20% |
| **Proactive outreach for fiber users** | Fiber optic + no tech support + high monthly charges | Address service quality concerns before they leave |
| **Payment method migration campaign** | Electronic check users | Migrate to auto-pay — data shows 28% lower churn rate |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Rohittt619/customer-churn-analysis
cd customer-churn-analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the `data/` folder.

**4. Run notebooks in order**
```bash
jupyter notebook
# Open: 01_EDA.ipynb → 02_Visualizations.ipynb → 03_ML_Models.ipynb
```

---

## 📦 Requirements

```
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
jupyter==1.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🧠 Key Learnings

- **Class imbalance** (73/27 split) significantly affects model performance — `class_weight='balanced'` and F1-score evaluation are essential
- **EDA revealed more insight** than the ML model alone — contract type and payment method were obvious from charts before any modelling
- **Recall matters more than precision** in churn use cases — it's better to flag a non-churner than to miss a real churner
- **Feature engineering opportunity:** combining `tenure` + `MonthlyCharges` into a `total_value_score` feature could improve model performance further

---

## 🔮 Future Improvements

- [ ] Apply SMOTE oversampling to handle class imbalance more aggressively
- [ ] Tune Gradient Boosting hyperparameters with `GridSearchCV`
- [ ] Add XGBoost and LightGBM for comparison
- [ ] Build a simple Flask API to score new customers in real-time
- [ ] Create a Power BI dashboard connected to model predictions

---

## 👨‍💻 Author

**Rohit Rathod**  
B.Tech in Information Technology (Data Science) — Ajeenkya DY Patil University  
📧 rrathod1101@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/rohit-rathod-19442a228) · [GitHub](https://github.com/Rohittt619)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*If this project helped you, consider giving it a ⭐ — it helps others find it!*
