import pandas as pd

p = "data/raw/secondary_data.csv"

# Read with delimiter
df = pd.read_csv(p, sep=";")

# Fix headers (replace hyphens with underscores)
df.columns = [c.strip().replace('-', '_') for c in df.columns]
df = df.rename(columns={"does_bruise_bleed": "does_bruise_or_bleed"})

print("✅ Fixed load | shape:", df.shape)
print("First 12 cols:", list(df.columns[:12]))

# Save back in standard CSV (comma-delimited)
df.to_csv(p, index=False)
