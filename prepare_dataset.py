import pandas as pd

# Load the raw dataset (it's tab separated, not comma separated)
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "text"])

# Save as clean CSV with correct column names
df.to_csv("spam_dataset.csv", index=False)

print("✅ Dataset saved as spam_dataset.csv")
print(df.head())
