from transformers import BertForSequenceClassification, BertTokenizer


def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize_for_classification(examples, tokenizer, max_len=128):
    tokens = tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=max_len
    )
    if "label" in examples:
        tokens["label"] = examples["label"]

    return tokens


def get_model(num_labels):
    return BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    )
