import pandas as pd
from pathlib import Path

files = {
    "Health": "outputs/health_psas_10000.csv",
    "Agriculture": "outputs/agriculture_10000.csv",
    "Education": "outputs/education_psas_10000.csv",
    "Security and Safety": "outputs/security_safety_10000.csv",
    "Governance": "outputs/governance_10000_final.csv",
}

dataframes = []

for domain, file_path in files.items():
    path = Path(file_path)

    if not path.exists():
        print(f"Missing file: {path}")
        continue

    df = pd.read_csv(path)

    # Add or overwrite the domain column
    df["domain"] = domain

    dataframes.append(df)

if not dataframes:
    raise FileNotFoundError("No domain CSV files were found.")

combined_df = pd.concat(dataframes, ignore_index=True)

# Remove exact duplicate PSA sentences, if any exist across domains
if "psa_text" in combined_df.columns:
    combined_df = combined_df.drop_duplicates(subset=["psa_text"])
elif "psa" in combined_df.columns:
    combined_df = combined_df.drop_duplicates(subset=["psa"])

# Create one new unique ID
combined_df.insert(
    0,
    "master_id",
    [f"PSA_MASTER_{i:05d}" for i in range(1, len(combined_df) + 1)]
)

output_path = "outputs/all_domains_50000.csv"
combined_df.to_csv(output_path, index=False)

print("\nCombination completed successfully.")
print("Total rows:", len(combined_df))
print("Output file:", output_path)

print("\nRows by domain:")
print(combined_df["domain"].value_counts())