import pandas as pd
import numpy as np
import json
import os
import torch
from torch.nn.functional import sigmoid
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from BERTmodel import get_model, get_tokenizer, tokenize_for_classification


def main(data_path = None):
    
    if data_path is None:
        base_dir = os.path.dirname(__file__)
        data_path = os.path.join(base_dir, 'data', 'binary', 'full.csv')
    
    df = pd.read_csv(data_path, sep='\t')
    # Just for testing we train the model just on 200 examples and not the whole dataset
    df = df.sample(n=200, random_state=42).reset_index(drop=True)
    
    texts = df["sentences"].tolist()
    labels =df[["ADR", "WD", "EF", "INF", "SSI", "DI"]].values.tolist()

    tokenizer = get_tokenizer()
    kf = KFold(n_splits=2, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
        print(f"\n--- Fold {fold + 1} ---")

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        train_dataset = Dataset.from_dict({"texts": train_texts, "labels": [list(map(float, lbl)) for lbl in train_labels]})
        val_dataset = Dataset.from_dict({"texts": val_texts, "labels": [list(map(float, lbl)) for lbl in val_labels]})
    
        train_dataset = train_dataset.map(lambda x: tokenize_for_classification(x, tokenizer), batched=True)
        val_dataset = val_dataset.map(lambda x: tokenize_for_classification(x, tokenizer), batched=True)

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        model = get_model(num_labels=6)

        training_args = TrainingArguments(
            output_dir=f"./results/fold_{fold + 1}",
            logging_dir=f"./logs/fold_{fold + 1}",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=1,
            weight_decay=0.01,
            save_strategy="epoch",
        )


        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            probs = sigmoid(torch.tensor(logits))
            preds = (probs > 0.5).long().cpu().numpy()
            labels = labels.astype(int)

            
            return {
                    "accuracy": accuracy_score(labels, preds), 
                    "precision_micro": precision_score(labels, preds, average="micro"),
                    "recall_micro": recall_score(labels, preds, average="micro"),
                    "f1_micro": f1_score(labels, preds, average="micro")
                    }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        print(f"fold {fold+1} completed")        
        metrics = trainer.evaluate()
        
        with open(f"./metrics/fold_{fold + 1}.json", "w") as f:
            json.dump(metrics, f)


        model.save_pretrained(f"./models/bert_model_fold_{fold + 1}")
        tokenizer.save_pretrained(f"./models/bert_model_fold_{fold + 1}")

if __name__ == "__main__":
    main()
