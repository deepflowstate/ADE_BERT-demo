# just a first outline how it could look. Please change or delete stuf as you like
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Load your dataset (replace with your file) --> here we ould either use PsyTAR or we could use ADE Corpus / we need to look into the dataset structure and fit the code to it
df = pd.read_csv("adr_sentences.csv")  # Expected columns: "text", "label"

# 2. Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2
)

# 3. Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(
    tokenize_function, batched=True
)
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(
    tokenize_function, batched=True
)

# 4. Load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 6. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 7. Train
trainer.train()

# 8. Save model
model.save_pretrained("./adr_bert_model")
tokenizer.save_pretrained("./adr_bert_model")
