"""
fix_pickle.py
Re-saves the model with pickle protocol 4 for compatibility.
"""

import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "multiclass_model.pkl")

print(f"Loading: {MODEL_PATH}")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print(f"✅ Loaded: {type(model).__name__}")

# Re-save with protocol 4 (works across Python 3.4–3.11)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f, protocol=4)
print(f"✅ Re-saved with protocol 4")

# Verify
with open(MODEL_PATH, "rb") as f:
    m2 = pickle.load(f)
print(f"✅ Verified: {type(m2).__name__}")
print("\n🎉 Done. Restart Streamlit.")