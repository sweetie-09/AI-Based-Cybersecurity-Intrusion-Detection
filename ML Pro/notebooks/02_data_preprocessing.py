import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Dataset path
file_path = "dataset/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv"

# Load dataset
df = pd.read_csv(file_path)

print("Original shape:", df.shape)

# Clean column names
df.columns = df.columns.str.strip()

print("\nColumn names cleaned.")

# Check missing values
print("\nTotal missing values:", df.isnull().sum().sum())

# Check infinite values
print(
    "Total infinite values:",
    np.isinf(df.select_dtypes(include=np.number)).sum().sum()
)

# Check duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Show original labels
print("\nOriginal labels:")
print(df["Label"].value_counts())


# --------------------------------------------------
# CLEANING
# --------------------------------------------------

# Step 1: Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

print("\nInfinite values replaced with NaN.")


# Step 2: Remove rows containing missing values
df.dropna(inplace=True)

print("Rows with missing values removed.")


# Step 3: Remove duplicate rows
df.drop_duplicates(inplace=True)

print("Duplicate rows removed.")


# --------------------------------------------------
# CREATE BINARY TARGET
# --------------------------------------------------

# BENIGN = 0
# Any attack = 1

df["Attack"] = df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)


# --------------------------------------------------
# FINAL INFORMATION
# --------------------------------------------------

print("\nCleaned shape:", df.shape)

print("\nFinal labels:")
print(df["Label"].value_counts())

print("\nBinary target distribution:")
print(df["Attack"].value_counts())

print("\nFinal missing values:", df.isnull().sum().sum())

print(
    "Final infinite values:",
    np.isinf(df.select_dtypes(include=np.number)).sum().sum()
)

print("Final duplicate rows:", df.duplicated().sum())

# --------------------------------------------------
# PREPARE FEATURES AND TARGET
# --------------------------------------------------

# Remove Label and Attack from input features
X = df.drop(columns=["Label", "Attack"])

# Attack is our target
y = df["Attack"]

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

print("\nNumber of features:", X.shape[1])

print("\nFeature data types:")
print(X.dtypes.value_counts())

print("\nFirst 5 target values:")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# --------------------------------------------------
# FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

print("Scaled training shape:", X_train_scaled.shape)
print("Scaled testing shape:", X_test_scaled.shape)

# --------------------------------------------------
# FIRST ML MODEL - LOGISTIC REGRESSION
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("\nLogistic Regression training completed.")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

y_pred = model.predict(X_test_scaled)

print("\nPrediction completed.")

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# DECISION TREE MODEL
# --------------------------------------------------

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

# Train
dt_model.fit(X_train, y_train)

# Predict
dt_pred = dt_model.predict(X_test)

# Evaluate
print("\nDecision Tree Performance:")

print("Accuracy :", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred, zero_division=0))
print("Recall   :", recall_score(y_test, dt_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, dt_pred, zero_division=0))

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

# --------------------------------------------------
# RANDOM FOREST MODEL
# --------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train
rf_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)

# Evaluate
print("\nRandom Forest Performance:")

print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred, zero_division=0))
print("Recall   :", recall_score(y_test, rf_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, rf_pred, zero_division=0))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

import joblib

joblib.dump(
    rf_model,
    "models/random_forest_model.pkl"
)

print("Model saved successfully!")