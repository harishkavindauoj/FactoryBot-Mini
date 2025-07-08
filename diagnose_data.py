"""
Diagnostic script to check data files and identify string data issues.
Run this before training to understand what's in your data files.
"""

import numpy as np
import os
import yaml


def load_config():
    """Load configuration file."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def diagnose_data_files():
    """Comprehensive data file diagnosis."""
    print("=" * 60)
    print("DATA FILE DIAGNOSTIC")
    print("=" * 60)

    # Load config
    try:
        config = load_config()
        print("✓ Config loaded successfully")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return

    # Check each data file
    files_to_check = [
        (config['data']['features_cache'], "Features (X)"),
        ("data/processed/risk_labels.npy", "Risk Labels (y_risk)"),
        ("data/processed/ttf_labels.npy", "TTF Labels (y_ttf)")
    ]

    for file_path, description in files_to_check:
        print(f"\n{'-' * 40}")
        print(f"CHECKING: {description}")
        print(f"File: {file_path}")
        print(f"{'-' * 40}")

        if not os.path.exists(file_path):
            print(f"✗ FILE NOT FOUND: {file_path}")
            continue

        try:
            # Try loading without allow_pickle first
            try:
                data = np.load(file_path)
                print("✓ Loaded without allow_pickle")
            except:
                data = np.load(file_path, allow_pickle=True)
                print("✓ Loaded with allow_pickle (contains Python objects)")

            # Basic info
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            print(f"Size: {data.size}")

            # Check data type category
            if data.dtype.kind in ['U', 'S']:
                print("⚠️  WARNING: Contains STRING data")
                print(f"String length: {data.dtype.itemsize}")
            elif data.dtype.kind == 'O':
                print("⚠️  WARNING: Contains OBJECT data (mixed types)")
            elif data.dtype.kind in ['i', 'u']:
                print("✓ Integer data")
            elif data.dtype.kind == 'f':
                print("✓ Float data")
            else:
                print(f"? Unknown data type kind: {data.dtype.kind}")

            # Show sample values
            flat_data = data.flatten()
            sample_size = min(10, len(flat_data))
            print(f"Sample values ({sample_size}/{len(flat_data)}):")

            for i in range(sample_size):
                value = flat_data[i]
                print(f"  [{i}]: {value} (type: {type(value).__name__})")

            # Check for problematic values
            if data.dtype == 'object':
                unique_types = set(type(x).__name__ for x in flat_data[:100])
                print(f"Types found in first 100 elements: {unique_types}")

            # Check for NaN/inf in numeric data
            if data.dtype.kind in ['f', 'i', 'u']:
                nan_count = np.sum(np.isnan(data))
                inf_count = np.sum(np.isinf(data))
                if nan_count > 0:
                    print(f"⚠️  NaN values: {nan_count}")
                if inf_count > 0:
                    print(f"⚠️  Infinite values: {inf_count}")

            # Try conversion test
            if data.dtype.kind in ['U', 'S', 'O']:
                print("\nTesting numeric conversion...")
                try:
                    if 'risk' in file_path.lower():
                        # Try label encoding for risk
                        unique_vals = np.unique(data)
                        print(f"Unique values: {unique_vals}")
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        numeric_data = le.fit_transform(data.flatten())
                        print(f"✓ Can convert to numeric: {unique_vals} -> {np.unique(numeric_data)}")
                    else:
                        # Try direct conversion
                        numeric_data = np.array(data, dtype=np.float32)
                        print("✓ Can convert to float32")
                except Exception as e:
                    print(f"✗ Cannot convert to numeric: {e}")

        except Exception as e:
            print(f"✗ Error loading file: {e}")

    print(f"\n{'=' * 60}")
    print("RECOMMENDATIONS:")
    print(f"{'=' * 60}")

    print("1. If any files show 'STRING' or 'OBJECT' warnings:")
    print("   - Check your data preprocessing pipeline")
    print("   - Ensure all features are numeric")
    print("   - Use label encoding for categorical variables")

    print("\n2. If risk labels are strings (high/medium/low):")
    print("   - The trainer will auto-convert them")
    print("   - Or preprocess them to 0/1/2 beforehand")

    print("\n3. If TTF values are strings:")
    print("   - Check your TTF calculation/extraction code")
    print("   - Ensure TTF values are numeric (hours/days)")

    print("\n4. If features contain mixed types:")
    print("   - Review your feature extraction process")
    print("   - Ensure all features are properly normalized")


if __name__ == "__main__":
    diagnose_data_files()