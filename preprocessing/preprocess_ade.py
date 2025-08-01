import sys
import os

# Add the parent directory to the Python path so local modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions for parsing and token processing
from utils.parse_indexes import parse_indexes
from preprocessing.tokenizer_utils import create_labels, get_tokenizer, label_map

import csv  # For reading the input ADE CSV file


def preprocess_ade_csv(input_csv_path, tokenizer, max_len=128):
    """
    Preprocesses the ADE dataset CSV for NER training.

    Tokenizes each text, extracts entity spans (DRUG and EFFECT), and assigns BIO labels.
    Pads/truncates sequences to `max_len`.

    Args:
        input_csv_path (str): Path to the ADE CSV file.
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
        max_len (int): Max sequence length (default: 128).

    Returns:
        List[Dict[str, List[int]]]: List of dictionaries with:
            - input_ids
            - attention_mask
            - labels (with -100 for padding)
    """
    encodings = []  # List to store processed examples

    # Open and read the CSV file
    with open(input_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # Iterate through each row in the CSV
        for row in reader:
            text = row["text"]
            indexes = parse_indexes(row["indexes"])  # Extract character spans for entities

            # Extract entity start and end positions
            drug_start_list = indexes["drug"]["start_char"]
            drug_end_list = indexes["drug"]["end_char"]
            effect_start_list = indexes["effect"]["start_char"]
            effect_end_list = indexes["effect"]["end_char"]

            # Skip row if any of the spans are missing
            if not drug_start_list or not drug_end_list or not effect_start_list or not effect_end_list:
                print(f"Skipping row due to empty spans: {row['text'][:30]}...")
                continue

            # Take only the first occurrence of each entity
            drug_start = drug_start_list[0]
            drug_end = drug_end_list[0]
            effect_start = effect_start_list[0]
            effect_end = effect_end_list[0]

            # Tokenize the text and get token-character alignment
            encoding = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_len)
            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
            offsets = encoding["offset_mapping"]

            # Remove special tokens ([CLS] and [SEP]) from tokens and offsets
            tokens = tokens[1:-1]
            offsets = offsets[1:-1]

            # Create BIO labels for the tokens based on entity spans
            label_ids = create_labels(text, (drug_start, drug_end), (effect_start, effect_end), tokens, offsets)

            # Also remove special tokens from input_ids and attention_mask
            input_ids = encoding["input_ids"][1:-1]
            attention_mask = encoding["attention_mask"][1:-1]

            # Pad input_ids, attention_mask, and labels to max_len
            pad_length = max_len - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_length
            attention_mask += [0] * pad_length
            label_ids += [-100] * pad_length  # -100 is ignored by most loss functions in token classification

            # Add the processed sample to the list
            encodings.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label_ids
            })

    return encodings  # Return all processed samples


if __name__ == "__main__":
    tokenizer = get_tokenizer(model_type="ner")
    ade_csv_path = os.path.join("data_sets", "ade_corpus_dataset", "ade_corpus_ner.csv")
    
    data = preprocess_ade_csv(ade_csv_path, tokenizer)
    
    print(f"Processed {len(data)} examples from the ADE dataset.")
    print(data[0])
    
    # Basic check for output structure
    assert len(data) > 0, "No examples were processed!"
    first_example = data[0]
    required_keys = {"input_ids", "attention_mask", "labels"}
    assert all(key in first_example for key in required_keys), "Some required keys are missing in the first example!"
    print("Basic test passed: data structure is correct.")
