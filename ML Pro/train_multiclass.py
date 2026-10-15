
import pandas as pd
import numpy as np
import glob
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Dataset folder
folder = "dataset/MachineLearningCVE"
files = glob.glob(os.path.join(folder, "*.csv"))

print("CSV files found:", len(files))

if not files:
    raise FileNotFoundError("No CSV files found!")

all_data = []

# 2. Read and preprocess all CSV files
for file in files:
    print("\nReading:", os.path.basename(file))

    for chunk in pd.read_csv(file, chunksize=100000):
        # Clean column names
        chunk.columns = chunk.columns.str.strip()

        if "Label" not in chunk.columns:
            continue

        # Clean labels
        chunk["Label"] = chunk["Label"].astype(str).str.strip()

        # Separate features and labels
        feature_columns = [
            col for col in chunk.columns if col != "Label"
        ]

        # Convert all features to numeric
        for col in feature_columns:
            chunk[col] = pd.to_numeric(
                chunk[col], errors="coerce"
            )

        # Replace infinite values
        chunk[feature_columns] = chunk[
            feature_columns
        ].replace([np.inf, -np.inf], np.nan)

        # Remove rows with missing labels
        chunk = chunk.dropna(subset=["Label"])

        # Sample up to 2000 rows per attack category
        for label, group in chunk.groupby("Label"):
            sampled = group.sample(
                n=min(len(group), 2000),
                random_state=42
            )
            all_data.append(sampled)

# 3. Combine all data
print("\nCombining datasets...")

df = pd.concat(all_data, ignore_index=True)

# Remove duplicate records
df = df.drop_duplicates()

print("\nCombined dataset shape:", df.shape)

print("\nAttack categories:")
print(df["Label"].value_counts())

# 4. Separate features and target
X = df.drop(columns=["Label"])
y = df["Label"]

# Ensure all feature values are numeric
X = X.replace([np.inf, -np.inf], np.nan)

# 5. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Create preprocessing + model pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# 7. Train the model
print("\nTraining improved multiclass model...")

model.fit(X_train, y_train)

# 8. Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
))

# 9. Save the improved model
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/multiclass_model.pkl"
)

print("\nImproved multiclass model saved successfully!")