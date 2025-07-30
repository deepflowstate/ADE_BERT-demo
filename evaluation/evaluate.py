import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import BertForSequenceClassification
from BERTmodel import get_tokenizer, tokenize_for_classification
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score


def main(data_path=None):
    base_dir = os.path.dirname(__file__)
    # hardcode model path for now...
    pretrained_path = os.path.join(base_dir, "models", "bert_model_fold_2_set_1")

    # use either pretrained model or vanilla BERT
    # TODO: make this a script param
    model = BertForSequenceClassification.from_pretrained(pretrained_path)
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    data_path = os.path.join(
        base_dir, "data_sets", "psytar_dataset", "PsyTAR_dataset.xlsx"
    )
    df = pd.read_excel(data_path, 3)

    texts = [str(x) for x in df["sentences"].tolist()]
    labels = [1 if x == 1.0 else 0 for x in df["ADR"].tolist()]

    tokenizer = get_tokenizer()
    val_dataset = Dataset.from_dict({"text": texts, "label": labels})
    tokenized_dataset = val_dataset.map(
        lambda x: tokenize_for_classification(x, tokenizer), batched=True
    )
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    val_dataloader = DataLoader(tokenized_dataset, batch_size=16)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="micro")

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1 (micro): {f1:.4f}")


if __name__ == "__main__":
    main()
