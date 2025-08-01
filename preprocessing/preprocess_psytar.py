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

            # Empty line indicates end of current example
            if not line:
                if tokens:
                    # Append the current example and reset buffers
                    examples.append({"tokens": tokens, "ner_tags": labels})
                    tokens = []
                    labels = []
                continue
            
            splits = line.split()
            if len(splits) >= 2:
                token = splits[0]
                label_str = splits[-1]
                tokens.append(token)
                # Map string label to numeric label, default to 0 if not found
                labels.append(label_map.get(label_str, 0))
        
        # Add last example if file doesn't end with a blank line
        if tokens:
            examples.append({"tokens": tokens, "ner_tags": labels})

    return examples



def preprocess_psytar_conll(input_conll_path: str, tokenizer, max_len=128):
    """
    Preprocesses the PsyTAR dataset from a CoNLL file format, producing tokenized inputs with labels.

    Args:
        input_conll_path (str): path to the psytar.conll file
        tokenizer: HuggingFace tokenizer
        max_len (int): maximum sequence length

    Returns:
        List[Dict]: list of dictionaries containing input_ids, attention_mask, and labels
    """
    examples = read_conll_file(input_conll_path)  # Read raw examples from the file
    encodings = []
    # Skip examples with empty tokens or labels (empty spans)
    for example in examples:
        if not example["tokens"] or not example["ner_tags"]:
            print(f"Skipping example due to empty tokens or labels: {example}")
            continue

        # Tokenize the example and map labels to tokenized tokens
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