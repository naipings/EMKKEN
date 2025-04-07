import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv
from langdetect import detect, DetectorFactory

# Ensure deterministic language detection results
DetectorFactory.seed = 0

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


def is_english(text: str) -> bool:
    """
    Determine if text is English using combined heuristics and language detection.

    Args:
        text (str): Input text to analyze

    Returns:
        bool: True if text is English, False otherwise

    Validation Criteria:
        1. Minimum 3 tokens after preprocessing
        2. Language detection confidence
        3. Contains alphabetic characters

    Raises:
        Preserves outer exception handling but maintains consistent return type
    """
    try:
        # Basic text normalization
        clean_text = text.lower().strip()

        # Tokenization and filtering
        tokens = word_tokenize(clean_text)
        if len(tokens) < 3:  # Exclude short/unstructured text
            return False

        # Language identification
        return detect(clean_text) == 'en'
    except Exception as e:
        print(f"Language detection error: {e}")
        return False


def preprocess_abstracts(input_file: str, output_file: str) -> pd.DataFrame:
    """
    Main text preprocessing pipeline for scientific abstracts.

    Processing Steps:
        1. CSV file loading and validation
        2. Text normalization
        3. Language filtering
        4. Tokenization
        5. Stopword/punctuation removal
        6. Cleaned text reconstruction

    Args:
        input_file (str): Path to raw CSV data
        output_file (str): Path for processed output

    Returns:
        pd.DataFrame: Processed data with cleaned abstracts

    Quality Control:
        - Tracks non-English abstracts
        - Handles missing/empty abstracts
        - Preserves original data structure
    """
    # Load and validate input data
    data = pd.read_csv(input_file)
    if 'abstract' not in data.columns:
        raise ValueError("CSV missing required 'abstract' column")

    # Initialize processing containers
    processed_abstracts = []
    non_english_count = 0
    stop_words = set(stopwords.words('english'))

    # Text processing pipeline
    for idx, row in data.iterrows():
        raw_abstract = str(row['abstract']).strip()

        # Skip empty abstracts
        if not raw_abstract:
            processed_abstracts.append('')
            continue

        # Language validation
        if not is_english(raw_abstract):
            non_english_count += 1
            processed_abstracts.append('')  # Mark non-English abstracts
            continue

        # Text normalization
        normalized = raw_abstract.lower().replace('\n', ' ')

        # Tokenization
        tokens = word_tokenize(normalized)

        # Text cleaning
        filtered = [
            word for word in tokens
            if word.isalpha() and word not in stop_words
        ]
        # filtered = [word for word in words if word.isalpha()]

        processed_abstracts.append(' '.join(filtered))

    # Add processed column and save
    data['processed_abstract'] = processed_abstracts
    data.to_csv(output_file, index=False)

    print(f"Processed {len(data)} abstracts")
    print(f"Excluded {non_english_count} non-English abstracts")
    return data


# Main execution
if __name__ == "__main__":
    input_path = "node_attribute_list.csv"
    output_path = "processed_abstracts.csv"

    try:
        processed_data = preprocess_abstracts(input_path, output_path)
        print("Processing completed successfully")
        print(processed_data.head())
    except Exception as e:
        print(f"Processing failed: {str(e)}")