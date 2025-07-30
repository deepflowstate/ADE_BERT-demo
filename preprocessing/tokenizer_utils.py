from transformers import BertTokenizer, BertTokenizerFast

label_list = ["O", "B-DRUG", "I-DRUG", "B-EFFECT", "I-EFFECT"]
label_map = {label: i for i, label in enumerate(label_list)}

def get_tokenizer(model_type="classification"):
    """
        Returns a pretrained Bert model depending on the type of task that we choose to perform: classification or name entity recognition    

        Args:
            model_type: string ("ner", "classification")
    """
    if model_type == "classification":
        return BertTokenizer.from_pretrained("bert-base-uncased")
    elif model_type == "ner":
        return BertTokenizerFast.from_pretrained("bert-base-uncased")
    else:
        raise ValueError(f"Unknown model_type {model_type}")

def tokenize_for_model(examples: dict, tokenizer, model_type="classification", max_len=128):
    """
    Tokenizes input data for text classification or Named Entity Recognition (NER) tasks.

    This function prepares input features for transformer-based models using a tokenizer. It supports
    both classification and token-level labeling tasks such as NER.

    Args:
        examples (dict): A dictionary containing the input data to tokenize. Expected keys depend on `model_type`:
            - For "classification": must contain the key "text" (a string or list of strings).
              Optionally, may contain "label" (an integer or list of integers).
            - For "ner": must contain "tokens" (a list of lists of strings) and "ner_tags" (a list of lists of integers).
        tokenizer (PreTrainedTokenizer): A HuggingFace tokenizer instance (e.g., from AutoTokenizer).
        model_type (str, optional): The type of task to prepare data for. Supported values:
            - "classification" (default): for text classification tasks.
            - "ner": for Named Entity Recognition tasks.
        max_len (int): Maximum token sequence length. Longer sequences will be truncated and shorter ones
            will be padded to this length. Default is 128.

    Returns:
        dict: A dictionary of tokenized inputs, including:
            - input_ids, attention_mask, and any other tokenizer outputs
            - labels: included only if label/ner_tags are provided

    Raises:
        ValueError: If `model_type` is not one of the supported values ("classification" or "ner").

    Notes:
        - In NER mode, special tokens and subword tokens are aligned using `-100` to be ignored by the loss function.
        - B-* labels are converted to I-* for subword tokens to maintain proper sequence tagging.
        - `return_offsets_mapping` is used internally to align word-level labels with token-level inputs.
    """
    #------- classification
    if model_type == "classification":
        tokens = tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=max_len
        )
        if "label" in examples:
            tokens["labels"] = examples["label"]
        return tokens
    # ----- name entity recognition
    elif model_type == "ner":
        # Tokenize at token-level
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
        )
        # word_ids associates each token with a number or none in the case of special characters (e.g., [CLS], [SEP])
        word_ids = tokenized_inputs.word_ids()

        labels = []
        previous_word_idx = None
        # Loop over tokens in each example (lines of text)
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)  # Ignore special characters (-100 is a special number for PyTorch)
            elif word_idx != previous_word_idx:
                labels.append(examples["ner_tags"][word_idx])
            else:
                # If it is subword token, convert its label to I-* (inside token)
                label = examples["ner_tags"][word_idx]
                if label % 2 == 1:  # even label = B-*
                    label += 1      # convert B- in I-
                labels.append(label)
            # Update previous_word_idx with the last id used
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    else:
        raise ValueError(f"Unknown model_type {model_type}")


def create_labels(text, drug_span, effect_span, tokens, offsets):
    labels = ["O"] * len(tokens)

    def mark_span(start_char, end_char, b_label, i_label):
        for i, (start, end) in enumerate(offsets):
            if end <= start_char:
                continue
            if start >= end_char:
                break
            # If the token intersecates the span
            if start_char <= start < end_char:
                if labels[i] == "O":
                    labels[i] = b_label
                else:
                    labels[i] = i_label

    mark_span(drug_span[0], drug_span[1], "B-DRUG", "I-DRUG")
    mark_span(effect_span[0], effect_span[1], "B-EFFECT", "I-EFFECT")

    return [label_map[label] for label in labels]