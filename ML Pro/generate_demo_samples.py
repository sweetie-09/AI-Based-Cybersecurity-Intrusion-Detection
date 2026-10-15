"""
generate_demo_samples.py

Samples real rows from the CICIDS2017 (MachineLearningCVE) dataset
and saves them as demo_samples.pkl for Demo Mode.
"""

import os
import glob
import pandas as pd
import pickle

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "MachineLearningCVE")
OUTPUT_PATH = os.path.join(MODELS_DIR, "demo_samples.pkl")

BENIGN_SAMPLES = 100
ATTACK_SAMPLES = 200
LABEL_COLUMN = "Label"

# ============================================================
# FIND CSVs
# ============================================================

def find_csvs():
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Dataset folder not found: {DATASET_DIR}")
        return []

    csvs = sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv")))

    if not csvs:
        print(f"❌ No CSVs found in {DATASET_DIR}")
        return []

    print(f"📂 Found {len(csvs)} CSV file(s):")
    for c in csvs:
        print(f"   - {os.path.basename(c)}")

    return csvs


# ============================================================
# LOAD AND COMBINE
# ============================================================

def load_all_csvs(csv_paths):
    frames = []

    for path in csv_paths:
        print(f"\n📖 Loading: {os.path.basename(path)}")
        try:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")

        df.columns = df.columns.str.strip()
        print(f"   Rows: {len(df):,}  Columns: {len(df.columns)}")

        if LABEL_COLUMN in df.columns:
            df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n✅ Combined: {len(combined):,} total rows")

    return combined


# ============================================================
# SAMPLING
# ============================================================

def sample_rows(df):
    benign_df = df[df[LABEL_COLUMN].str.upper() == "BENIGN"].copy()
    attack_df = df[df[LABEL_COLUMN].str.upper() != "BENIGN"].copy()

    print(f"\n📊 Breakdown:")
    print(f"   Benign:  {len(benign_df):,}")
    print(f"   Attacks: {len(attack_df):,}")

    if len(benign_df) == 0 or len(attack_df) == 0:
        print("❌ Missing benign or attack samples.")
        return None, None

    print(f"\n🎯 Attack types available:")
    for label, count in attack_df[LABEL_COLUMN].value_counts().items():
        print(f"   {label}: {count:,}")

    n_benign = min(BENIGN_SAMPLES, len(benign_df))
    benign_samples = benign_df.sample(n=n_benign, random_state=42)
    print(f"\n✅ Sampled {n_benign} benign rows")

    attack_types = attack_df[LABEL_COLUMN].unique()
    per_type = max(1, ATTACK_SAMPLES // len(attack_types))
    attack_sample_list = []

    for atype in attack_types:
        type_df = attack_df[attack_df[LABEL_COLUMN] == atype]
        n = min(per_type, len(type_df))
        attack_sample_list.append(type_df.sample(n=n, random_state=42))
        print(f"   {atype}: sampled {n}")

    attack_samples = pd.concat(attack_sample_list, ignore_index=True)
    print(f"\n✅ Sampled {len(attack_samples)} attack rows total")

    return benign_samples, attack_samples


# ============================================================
# CONVERT TO FLOW LIST
# ============================================================

def convert_to_flow_list(samples_df, feature_cols):
    flows = []
    for _, row in samples_df.iterrows():
        flow = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
        flows.append(flow)
    return flows


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("DEMO SAMPLES GENERATOR — CICIDS2017 (MachineLearningCVE)")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    csvs = find_csvs()
    if not csvs:
        return

    df = load_all_csvs(csvs)

    if LABEL_COLUMN not in df.columns:
        print(f"❌ Label column '{LABEL_COLUMN}' not found.")
        print(f"   Columns available: {list(df.columns)}")
        return

    benign_df, attack_df = sample_rows(df)
    if benign_df is None:
        return

    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]
    print(f"\n🔧 Using {len(feature_cols)} feature columns")

    benign_flows = convert_to_flow_list(benign_df, feature_cols)
    attack_flows = convert_to_flow_list(attack_df, feature_cols)

    bundle = {
        "benign": benign_flows,
        "attacks": attack_flows,
        "feature_columns": feature_cols,
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n✅ Saved: {len(benign_flows)} benign + {len(attack_flows)} attack samples")
    print(f"   → {OUTPUT_PATH}")
    print(f"\n🎉 Demo Mode ready!")


if __name__ == "__main__":
    main()