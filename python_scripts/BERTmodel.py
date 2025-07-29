import csv
import ast
from transformers import (
    BertForSequenceClassification,
    BertForTokenClassification,
    BertTokenizer,
    BertTokenizerFast,
)
from datasets import Dataset
import re

# Lista label BIO per NER e mappatura
label_list = ["O", "B-DRUG", "I-DRUG", "B-EFFECT", "I-EFFECT"]
label_map = {label: i for i, label in enumerate(label_list)}


def get_tokenizer(model_type="classification"):
    if model_type == "classification":
        return BertTokenizer.from_pretrained("bert-base-uncased")
    elif model_type == "ner":
        return BertTokenizerFast.from_pretrained("bert-base-uncased")
    else:
        raise ValueError(f"Unknown model_type {model_type}")


def tokenize(examples, tokenizer, model_type="classification", max_len=128):
    if model_type == "classification":
        tokens = tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=max_len
        )
        if "label" in examples:
            tokens["labels"] = examples["label"]
        return tokens

    elif model_type == "ner":
        # Tokenizza e allinea label token-level
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
        )
        word_ids = tokenized_inputs.word_ids()

        labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)  # Ignora i token speciali
            elif word_idx != previous_word_idx:
                labels.append(examples["ner_tags"][word_idx])
            else:
                # Se è subword token, usa etichetta I- (se B- cambia in I-)
                label = examples["ner_tags"][word_idx]
                if label % 2 == 1:  # label dispari = B-*
                    label += 1      # converte B- in I-
                labels.append(label)
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    else:
        raise ValueError(f"Unknown model_type {model_type}")


def get_model(num_labels, model_type="classification"):
    if model_type == "classification":
        return BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )
    elif model_type == "ner":
        return BertForTokenClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )
    else:
        raise ValueError(f"Unknown model_type {model_type}")


def parse_indexes(indexes_str):
    cleaned = re.sub(r"array\((\[.*?\])(?:, dtype=\w+)?\)", r"\1", indexes_str)
    cleaned = re.sub(r",?\s*dtype=\w+", "", cleaned)
    cleaned = cleaned.replace(")", "")
    return ast.literal_eval(cleaned)


def create_labels(text, drug_span, effect_span, tokens, offsets):
    labels = ["O"] * len(tokens)

    def mark_span(start_char, end_char, b_label, i_label):
        for i, (start, end) in enumerate(offsets):
            if end <= start_char:
                continue
            if start >= end_char:
                break
            # Se token interseca la span
            if start_char <= start < end_char:
                if labels[i] == "O":
                    labels[i] = b_label
                else:
                    labels[i] = i_label

    mark_span(drug_span[0], drug_span[1], "B-DRUG", "I-DRUG")
    mark_span(effect_span[0], effect_span[1], "B-EFFECT", "I-EFFECT")

    return [label_map[label] for label in labels]


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
    # Esempio uso

    model_type = "ner"  # o "classification"
    tokenizer = get_tokenizer(model_type=model_type)

    if model_type == "ner":
        input_csv_path = "data_sets/ade_corpus_dataset/ade_corpus_ner.csv"  # percorso al tuo csv ADE
        max_len = 128

        dataset_encodings = preprocess_ade_csv(input_csv_path, tokenizer, max_len)

        # Converte lista dict in Dataset HuggingFace
        dataset = Dataset.from_list(dataset_encodings)

        print(dataset[0])

        model = get_model(num_labels=len(label_list), model_type=model_type)

    else:
        # Per classificazione: esempio minimo
        examples = {"text": "Questo è un esempio.", "label": 1}
        tokens = tokenize(examples, tokenizer, model_type=model_type)
        print(tokens)
        model = get_model(num_labels=2, model_type=model_type)
