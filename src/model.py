import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/figures", exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load data
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Remove customerID
df.drop(
    "customerID",
    axis=1,
    inplace=True
)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Convert categorical columns
df = pd.get_dummies(
    df,
    drop_first=True
)

print(df.shape)

# Features and target
X = df.drop(
    "Churn_Yes",
    axis=1
)

y = df["Churn_Yes"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Random Forest":
        RandomForestClassifier(random_state=42),

    "Gradient Boosting":
        GradientBoostingClassifier(random_state=42)
}

for name, model in models.items():

    print(f"\nRunning {name}...")

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print(classification_report(y_test, y_pred))

    print(
        "ROC AUC:",
        roc_auc_score(
            y_test,
            model.predict_proba(X_test_scaled)[:,1]
        )
    )
    
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





#
results = []

for name, model in models.items():

    print(f"\nRunning {name}...")

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    roc = roc_auc_score(
        y_test,
        model.predict_proba(X_test_scaled)[:, 1]
    )

    # Find the positive class automatically
    class_keys = [
        k for k in report.keys()
        if k not in ["accuracy", "macro avg", "weighted avg"]
    ]

    positive_class = class_keys[-1]

    results.append({
        "Model": name,
        "Accuracy": report["accuracy"],
        "Precision": report[positive_class]["precision"],
        "Recall": report[positive_class]["recall"],
        "F1_Score": report[positive_class]["f1-score"],
        "ROC_AUC": roc
    })

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df)

results_df.to_csv(
    "outputs/model_results.csv",
    index=False
)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/model_results.csv",
    index=False
)

# Sort models by ROC-AUC
results_df = results_df.sort_values(
    by="ROC_AUC",
    ascending=False
)

print("\nModel Comparison:")
print(results_df)

print("\nBest Model:")
print(results_df.iloc[0])

print("Model Results Saved")

