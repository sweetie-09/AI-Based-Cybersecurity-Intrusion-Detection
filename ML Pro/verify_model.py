import joblib

MODEL_PATH = "models/multiclass_model.pkl"

print("Loading model...")

model = joblib.load(MODEL_PATH)

print("\nModel loaded successfully!")
print("Model type:", type(model).__name__)

features = model.feature_names_in_

print("\nNumber of features:", len(features))

print("\nFeature names:")
for i, feature in enumerate(features, start=1):
    print(f"{i}. {feature}")

print("\nNumber of classes:", len(model.classes_))

print("\nClasses:")
for i, cls in enumerate(model.classes_, start=1):
    print(f"{i}. {cls}")