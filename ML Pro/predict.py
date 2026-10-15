
import pandas as pd
import numpy as np
import joblib
import glob
import os

# 1. Load the improved multiclass model
model = joblib.load("models/multiclass_model.pkl")

print("Multiclass model loaded successfully!")

# Get the exact feature names and order used during training
feature_columns = list(model.feature_names_in_)

print("Number of features:", len(feature_columns))
print("Number of attack categories:", len(model.classes_))

# 2. Find all dataset CSV files
folder = "dataset/MachineLearningCSV/MachineLearningCVE"
files = glob.glob(os.path.join(folder, "*.csv"))

print("\nCSV files found:", len(files))

# 3. Read and preprocess records
all_data = []

for file in files:
    print("\nReading:", os.path.basename(file))

    for chunk in pd.read_csv(file, chunksize=100000):

        # Clean column names
        chunk.columns = chunk.columns.str.strip()

        if "Label" not in chunk.columns:
            continue

        # Clean labels
        chunk["Label"] = chunk["Label"].astype(str).str.strip()

        # Ensure all model features are present
        for col in feature_columns:
            if col not in chunk.columns:
                chunk[col] = np.nan

        # Select features in exactly the training order
        X_chunk = chunk[feature_columns].copy()

        # Convert features to numeric
        for col in feature_columns:
            X_chunk[col] = pd.to_numeric(
                X_chunk[col], errors="coerce"
            )

        # Replace infinite values
        X_chunk = X_chunk.replace(
            [np.inf, -np.inf], np.nan
        )

        # Combine features with actual labels
        cleaned = X_chunk.copy()
        cleaned["Label"] = chunk["Label"]

        # Remove rows with missing labels only.
        # The saved pipeline imputes missing feature values.
        cleaned = cleaned.dropna(subset=["Label"])

        # Select up to 2 records per category per chunk
        for label, group in cleaned.groupby("Label"):

            sampled = group.sample(
                n=min(len(group), 2),
                random_state=42
            )

            all_data.append(sampled)

# 4. Combine all records
print("\nCombining records...")

if not all_data:
    raise ValueError("No records were found!")

df = pd.concat(all_data, ignore_index=True)
df = df.drop_duplicates()

print("Total sampled records:", len(df))

# 5. Separate features and actual labels
X = df[feature_columns].copy()
actual_labels = df["Label"].reset_index(drop=True)

# 6. Predict using the saved preprocessing + model pipeline
print("\nPredicting attack categories...")

predictions = model.predict(X)

# 7. Display predictions
correct = 0

for i in range(len(predictions)):

    actual = actual_labels.iloc[i]
    predicted = predictions[i]

    print("\nActual:", actual)
    print("Predicted:", predicted)

    if actual == predicted:
        print("Result: Correct")
        correct += 1
    else:
        print("Result: Incorrect")

# 8. Summary
total = len(predictions)
incorrect = total - correct

print("\n========== SUMMARY ==========")

print("Total records:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", incorrect)

if total > 0:
    accuracy = correct / total * 100
    print(f"Sample accuracy: {accuracy:.2f}%")

print("\nCategories tested:")
print(actual_labels.value_counts())

# 9. Save detailed predictions to CSV
results = pd.DataFrame({
    "Actual": actual_labels,
    "Predicted": predictions,
    "Correct": actual_labels == predictions
})

os.makedirs("results", exist_ok=True)

results.to_csv(
    "results/prediction_results.csv",
    index=False
)

print("\nDetailed predictions saved to:")
print("results/prediction_results.csv")