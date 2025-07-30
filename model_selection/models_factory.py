from transformers import BertForSequenceClassification, BertForTokenClassification

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