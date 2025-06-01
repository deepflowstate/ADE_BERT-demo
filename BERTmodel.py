
from transformers import BertForSequenceClassification, BertTokenizer

def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_for_classification(examples, tokenizer, max_len=128):
    tokens = tokenizer(
                    examples["texts"],
                    padding="max_length",
                    truncation=True, max_length=max_len
                    )
    if "labels" in examples:
        tokens["labels"] = examples["labels"]
        
    return tokens
    
def get_model(num_labels):
    return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels, problem_type="multi_label_classification")