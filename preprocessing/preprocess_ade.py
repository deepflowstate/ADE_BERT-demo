import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.parse_indexes import parse_indexes
from preprocessing.tokenizer_utils import create_labels, get_tokenizer, label_map
import csv

def preprocess_ade_csv(input_csv_path, tokenizer, max_len=128):
    encodings = []

    with open(input_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row["text"]
            indexes = parse_indexes(row["indexes"])

            # Controllo su start_char per drug ed effect
            drug_start_list = indexes["drug"]["start_char"]
            drug_end_list = indexes["drug"]["end_char"]
            effect_start_list = indexes["effect"]["start_char"]
            effect_end_list = indexes["effect"]["end_char"]

            # Se vuoti, skippa la riga o gestisci diversamente
            if not drug_start_list or not drug_end_list or not effect_start_list or not effect_end_list:
                print(f"Skipping row due to empty spans: {row['text'][:30]}...")
                continue

            drug_start = drug_start_list[0]
            drug_end = drug_end_list[0]
            effect_start = effect_start_list[0]
            effect_end = effect_end_list[0]

            encoding = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_len)
            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
            offsets = encoding["offset_mapping"]

            tokens = tokens[1:-1]
            offsets = offsets[1:-1]

            label_ids = create_labels(text, (drug_start, drug_end), (effect_start, effect_end), tokens, offsets)

            input_ids = encoding["input_ids"][1:-1]
            attention_mask = encoding["attention_mask"][1:-1]

            pad_length = max_len - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_length
            attention_mask += [0] * pad_length
            label_ids += [-100] * pad_length

            encodings.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label_ids
            })

    return encodings

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
