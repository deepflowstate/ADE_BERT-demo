import os
from typing import List, Dict
from .tokenizer_utils import tokenize_for_model, label_map, get_tokenizer

def read_conll_file(file_path: str) -> List[Dict]:
    """
        Reads a file in CoNLL format and converts it into a list of examples with tokens and ner_tags.

        Args:
            file_path (str): path to the .conll file

        Returns:
            List[Dict]: list of examples, each example is a dict {"tokens": [...], "ner_tags": [...]}
    """

    examples = []
    tokens = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    examples.append({"tokens": tokens, "ner_tags": labels})
                    tokens = []
                    labels = []
                continue
            
            splits = line.split()
            if len(splits) >= 2:
                token = splits[0]
                label_str = splits[-1]
                tokens.append(token)
                labels.append(label_map.get(label_str, 0))
            
        if tokens:
            examples.append({"tokens": tokens, "ner_tags": labels})

    return examples


def preprocess_psytar_conll(input_conll_path: str, tokenizer, max_len=128):
    """
    Preprocessing del dataset PsyTAR da file .conll, produce tokenizzazioni con label

    Args:
        input_conll_path (str): path al file psytar.conll
        tokenizer: tokenizer di Huggingface
        max_len (int): lunghezza max sequenze

    Returns:
        List[Dict]: lista di dizionari contenenti input_ids, attention_mask e labels
    """
    examples = read_conll_file(input_conll_path)
    encodings = []

    for example in examples:
        # Skip examples with empty tokens or labels (empty spans)
        if not example["tokens"] or not example["ner_tags"]:
            print(f"Skipping example due to empty tokens or labels: {example}")
            continue

        tokenized = tokenize_for_model(example, tokenizer, model_type="ner", max_len=max_len)
        encodings.append(tokenized)
    
    return encodings

if __name__ == "__main__":
    tokenizer = get_tokenizer(model_type="ner")
    conll_path = os.path.join("data_sets", "psytar_dataset", "psytar_ner.txt")

    data = preprocess_psytar_conll(conll_path, tokenizer)

    print(f"Processed {len(data)} examples from PsyTAR dataset.")
    print(data[0])

    assert len(data) > 0, "No examples were processed!"
    first_example = data[0]
    required_keys = {"input_ids", "attention_mask", "labels"}
    assert all(key in first_example for key in required_keys), "Some required keys are missing in the first example!"
    print("Basic test passed: data structure is correct.")