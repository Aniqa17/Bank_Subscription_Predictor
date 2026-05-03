# Bank Marketing — Subscription Prediction

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_style('whitegrid')


# 1. DATA CLEANING

# Load the data
df = pd.read_csv('bank_detailed.csv')
print("Dataset shape:", df.shape)
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Remove duplicate rows
initial_rows = df.shape[0]
df = df.drop_duplicates()
print(f"\nRemoved {initial_rows - df.shape[0]} duplicate rows.")

# Convert target variable to numeric  (no → 0, yes → 1)
df['subscribed'] = df['subscribed'].map({'no': 0, 'yes': 1})
print("\nTarget distribution:")
print(df['subscribed'].value_counts())

# Convert other binary columns to numeric
binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    df[col] = df[col].map({'no': 0, 'yes': 1}).fillna(0)
    print(f"NaN in '{col}' after conversion: {df[col].isnull().sum()}")


# 2. EXPLORATORY DATA ANALYSIS – TWO CLEAN FIGURES (NO OVERLAP)

num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

# FIGURE 1: Target Distribution + Correlation Matrix
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle('Bank Marketing – Overview', fontsize=16, fontweight='bold')

# Target distribution
counts = df['subscribed'].value_counts()
ax1.bar(['No (0)', 'Yes (1)'], counts.values, color=['steelblue', 'coral'], edgecolor='black')
ax1.set_title('Subscription Distribution')
ax1.set_ylabel('Count')
for i, v in enumerate(counts.values):
    ax1.text(i, v + 200, str(v), ha='center', fontweight='bold')

# Correlation heatmap
corr = df[num_cols + ['subscribed']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax2, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
ax2.set_title('Correlation Matrix')

plt.tight_layout()
plt.savefig('dashboard_overview.png', dpi=150, bbox_inches='tight')
print("Saved: dashboard_overview.png")
plt.show()

# FIGURE 2: All Numerical Distributions (3x2 grid) 
fig2, axes = plt.subplots(3, 2, figsize=(12, 10))
fig2.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold')
axes = axes.flatten()

colors = ['#4C72B0', '#DD8452', "#549663", '#C44E52', '#8172B3', '#937860']
for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=30, color=colors[i], edgecolor='black', alpha=0.8)
    axes[i].set_title(col, fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Hide any unused subplot (not needed because 3x2 = 6 exactly)
plt.tight_layout()
plt.savefig('dashboard_distributions.png', dpi=150, bbox_inches='tight')
print("Saved: dashboard_distributions.png")
plt.show()

# 3. DATA PREPARATION FOR MODELLING

# Encode categorical variables with LabelEncoder
label_encoders = {}
cat_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop('subscribed', axis=1)
y = df['subscribed']

# Train / test split  (stratified to keep class balance the same in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set size : {X_train.shape[0]}")
print(f"Test set size     : {X_test.shape[0]}")
print("NaN in X_train before scaling:", X_train.isnull().sum().sum())

# Standardise numerical features
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])


# 4. MODEL TRAINING

print("\nTraining Logistic Regression …")
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
print("Training complete.")


# 5. PREDICTION & EVALUATION

y_pred = model.predict(X_test)

accuracy    = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_rep   = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_rep)


# Feature coefficients
coeff_df = pd.DataFrame({
    'feature'    : X.columns,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', ascending=False)

print("\nTop features pushing TOWARDS subscription:")
print(coeff_df.head(5).to_string(index=False))
print("\nTop features pushing AWAY from subscription:")
print(coeff_df.tail(5).to_string(index=False))


# 6. PREDICT FOR A NEW CUSTOMER

def predict_customer(customer_data):
    """
    Takes a dictionary of customer features and returns:
      - prediction : 1 (will subscribe) or 0 (will not subscribe)
      - probability: probability of subscribing (0.0 – 1.0)
    """
    processed = {}
    for col in X.columns:
        val = customer_data[col]
        if col in label_encoders:
            le = label_encoders[col]
            processed[col] = le.transform([val])[0] if val in le.classes_ else -1
        elif col in num_cols:
            processed[col] = val
        else:
            processed[col] = 1 if val == 'yes' else 0   # binary columns

    new_df = pd.DataFrame([processed])
    new_df[num_cols] = scaler.transform(new_df[num_cols])

    prediction  = model.predict(new_df)[0]
    probability = model.predict_proba(new_df)[0, 1]
    return prediction, probability


# Example customer
test_customer = {
    'age'      : 35,
    'job'      : 'management',
    'marital'  : 'married',
    'education': 'tertiary',
    'default'  : 'no',
    'balance'  : 1500,
    'housing'  : 'yes',
    'loan'     : 'no',
    'contact'  : 'cellular',
    'day'      : 15,
    'month'    : 'jun',
    'duration' : 400,    # ← longer calls → more likely to subscribe
    'campaign' : 1,
    'pdays'    : 10,
    'previous' : 2,
    'poutcome' : 'success'
}

pred, prob = predict_customer(test_customer)
print(f"\n── New Customer Prediction ──────────────────────────────")
print(f"Will subscribe? {'Yes' if pred == 1 else ' No'}  (probability: {prob:.2f})")