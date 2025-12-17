"""
Quick test script for transform.py
Run this from your project root:  python3 test_transform.py
"""

import os
import pandas as pd
from src.transform import transform_data, MushroomTransformer

# Path to your dataset (adjust if stored elsewhere)
DATA_PATH = os.path.join("data", "raw", "secondary_data.csv")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at {DATA_PATH}")
        return

    print("✅ Loading raw data...")
    df = pd.read_csv(DATA_PATH)
    print("Raw shape:", df.shape)

    # Run default transformation
    print("\n⚙️ Running default transform_data...")
    df_out = transform_data(df)
    print("Transformed shape:", df_out.shape)
    print(df_out.head())

    # Run with custom parameters (just to test class API)
    print("\n⚙️ Running with MushroomTransformer (rare_threshold=800)...")
    tx = MushroomTransformer(rare_threshold=800, z_thresh=2.7, seed=123)
    df_out2 = tx.transform(df)
    print("Custom transformed shape:", df_out2.shape)

    print("\n✅ Test complete. Logs written to logs/transform.log")

if __name__ == "__main__":
    main()
