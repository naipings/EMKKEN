import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize_features(file_paths: list, columns_to_scale: list) -> None:
    """
    Standardize specified numeric features across multiple CSV files.

    Args:
        file_paths (list): List of CSV file paths to process
        columns_to_scale (list): Column names to standardize

    Processing Pipeline:
        1. Load data from all files
        2. Combine target columns into single DataFrame
        3. Fit standardization scaler on combined data
        4. Apply transformation to each file
        5. Save standardized data back to original files

    Note:
        - Maintains original file structure and non-scaled columns
        - Handles missing target columns gracefully
        - Preserves row order during transformation
    """
    # Initialize storage for combined data
    all_data = pd.DataFrame()
    row_counts = []

    # Phase 1: Data Collection
    for path in file_paths:
        df = pd.read_csv(path)

        # Skip files missing required columns
        if not set(columns_to_scale).issubset(df.columns):
            print(f"Skipping {path} - missing required columns")
            continue

        # Track row counts for proper data partitioning
        row_counts.append(len(df))
        all_data = pd.concat([all_data, df[columns_to_scale]], ignore_index=True)

    # Phase 2: Standardization
    if len(all_data) > 0:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(all_data)
        scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)

        # Phase 3: Data Distribution
        start_idx = 0
        for path, row_count in zip(file_paths, row_counts):
            df = pd.read_csv(path)

            # Skip files that were excluded during collection
            if not set(columns_to_scale).issubset(df.columns):
                continue

            # Apply standardized values
            end_idx = start_idx + row_count
            df[columns_to_scale] = scaled_df.iloc[start_idx:end_idx].values
            start_idx = end_idx

            # Save updated data
            df.to_csv(path, index=False)
            print(f"Processed {path} - {row_count} rows standardized")
    else:
        print("No valid data found for standardization")


# Configuration
DATA_FILES = [
    "meta_data.csv",
    # Add additional file paths as needed
]

FEATURES_TO_SCALE = [
    'Wos_citations',
    'Indegree'
    # Add other numeric columns requiring standardization
]

# Execute standardization pipeline
standardize_features(DATA_FILES, FEATURES_TO_SCALE)
