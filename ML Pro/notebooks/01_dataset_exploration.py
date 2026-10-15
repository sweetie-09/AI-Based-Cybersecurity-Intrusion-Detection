import pandas as pd

file_path = "dataset/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv"

# Read only the Label column
labels = pd.read_csv(
    file_path,
    usecols=[" Label"]
)

print("Total rows:", len(labels))

print("\nNumber of unique labels:")
print(labels[" Label"].nunique())

print("\nLabel distribution:")
print(labels[" Label"].value_counts())