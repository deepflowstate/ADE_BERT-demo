from transformers import BertForSequenceClassification, BertForTokenClassification

def get_model(num_labels, model_type="classification"):
    """
    Loads a pre-trained BERT model for classification or NER.

    Args:
        num_labels (int): Number of output labels or classes.
        model_type (str): Either "classification" or "ner".

    Returns:
        PreTrainedModel: BERT model ready for fine-tuning.

    Raises:
        ValueError: If an unknown `model_type` is passed.
    """
    if model_type == "classification":
        # Load a BERT model for sequence classification
        return BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )
    elif model_type == "ner":
        # Load a BERT model for token classification (e.g., NER)
        return BertForTokenClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )
    else:
        # Raise an error for unsupported model types
        raise ValueError(f"Unknown model_type {model_type}")