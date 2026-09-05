import os
import pandas as pd


def clean_dataset(dataset_dir="dataset"):
    """
    Compare Test.csv against Meta.csv, delete any Test image files whose
    ClassId does not appear in Meta.csv, and write a cleaned Test.csv.

    Expected layout (script run from the folder that CONTAINS dataset_dir):
        dataset_dir/
            Meta.csv
            Test.csv
            Test/00000.png
            Test/00001.png
            ...
    """
    print(f"Analyzing dataset in: {dataset_dir}")

    test_csv = os.path.join(dataset_dir, "Test.csv")
    meta_csv = os.path.join(dataset_dir, "Meta.csv")

    if not all(os.path.exists(p) for p in [test_csv, meta_csv]):
        print("Error: Missing CSV files. Please ensure Test.csv and Meta.csv exist in the dataset folder.")
        return

    # Load CSVs
    test_df = pd.read_csv(test_csv)
    meta_df = pd.read_csv(meta_csv)

    # Valid ClassIds = whatever classes exist in Meta.csv
    valid_classes = set(meta_df['ClassId'].unique())
    print(f"Total valid classes found in Meta.csv: {len(valid_classes)}")
    print(f"Valid ClassIds: {sorted(valid_classes)}")

    # Find invalid entries in Test (ClassId not present in Meta.csv)
    invalid_mask = ~test_df['ClassId'].isin(valid_classes)
    invalid_df = test_df[invalid_mask]

    print(f"Found {len(invalid_df)} images in Test.csv with ClassIds not in Meta.csv.")

    # Remove physical files
    removed_count = 0
    missing_count = 0
    for _, row in invalid_df.iterrows():
        # row['Path'] looks like 'Test/00000.png' -> resolve relative to dataset_dir
        img_path = os.path.join(dataset_dir, row['Path'])

        if os.path.exists(img_path):
            try:
                os.remove(img_path)
                removed_count += 1
            except Exception as e:
                print(f"Could not remove {img_path}: {e}")
        else:
            missing_count += 1

    print(f"Successfully deleted {removed_count} invalid image files.")
    if missing_count:
        print(f"Note: {missing_count} listed files were not found on disk (already missing / path mismatch).")

    # Create new cleaned Test.csv (only rows whose ClassId is valid)
    cleaned_test_df = test_df[~invalid_mask]
    cleaned_csv_path = os.path.join(dataset_dir, "Test_cleaned.csv")
    cleaned_test_df.to_csv(cleaned_csv_path, index=False)

    print(f"Cleaned Test dataset info saved to: {cleaned_csv_path}")
    print(f"Rows kept: {len(cleaned_test_df)} / {len(test_df)}")
    print("Done! Review Test_cleaned.csv, then rename it to Test.csv once you're satisfied.")


if __name__ == "__main__":
    # Run this script from the folder that CONTAINS the 'dataset' folder,
    # i.e. dataset/Meta.csv, dataset/Test.csv, dataset/Test/*.png must exist.
    clean_dataset("dataset")