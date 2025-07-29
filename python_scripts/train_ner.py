import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import KFold
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from BERTmodel import get_model, get_tokenizer, preprocess_ade_csv

label_list = ["O", "B-DRUG", "I-DRUG", "B-EFFECT", "I-EFFECT"]  # Le tue label NER

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label_list[l] for l in label if l != -100] for label in labels
    ]
    true_preds = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }
    results.update(classification_report(true_labels, true_preds, output_dict=True))
    return results

def main(data_path=None):
    if data_path is None:
        base_dir = os.path.dirname(__file__)
        data_path = os.path.join(
            base_dir, "..", "data_sets", "ade_corpus_dataset", "ade_corpus_ner.csv"
        )

    max_len = 128

    # Parametri da testare
    param_grid = [
        {"learning_rate": 5e-5, "weight_decay": 0.01, "batch_size": 8},
        {"learning_rate": 3e-5, "weight_decay": 0.01, "batch_size": 16},
        {"learning_rate": 2e-5, "weight_decay": 0.0, "batch_size": 8},
    ]

    # Carica e preprocessa dati (lista di dict con input_ids, attention_mask, labels)
    dataset_encodings = preprocess_ade_csv(data_path, get_tokenizer(model_type="ner"), max_len=max_len)
    dataset = Dataset.from_list(dataset_encodings)

    # KFold (senza stratify perché NER)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    all_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n--- Fold {fold + 1} ---")

        train_dataset = dataset.select(train_idx)
        val_dataset = dataset.select(val_idx)

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        for i, params in enumerate(param_grid):
            print(f"Fold {fold + 1}, Set {i + 1} - Params: {params}")

            model = get_model(num_labels=len(label_list), model_type="ner")

            training_args = TrainingArguments(
                output_dir=f"./results/fold_{fold + 1}/set_{i + 1}",
                logging_dir=f"./logs/fold_{fold + 1}/set_{i + 1}",
                per_device_train_batch_size=params["batch_size"],
                per_device_eval_batch_size=params["batch_size"],
                num_train_epochs=3,
                learning_rate=params["learning_rate"],
                weight_decay=params["weight_decay"],
                save_total_limit=2,
                metric_for_best_model="f1",
                greater_is_better=True,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
            )

            trainer.train()
            metrics = trainer.evaluate()

            os.makedirs("./metrics", exist_ok=True)
            with open(f"./metrics/fold_{fold + 1}_set_{i + 1}.json", "w") as f:
                json.dump(metrics, f)

            os.makedirs("./models", exist_ok=True)
            model.save_pretrained(f"./models/fold_{fold + 1}_set_{i + 1}")
            trainer.tokenizer.save_pretrained(f"./models/fold_{fold + 1}_set_{i + 1}")

            all_results.append(
                {
                    "fold": fold + 1,
                    "param_set": i + 1,
                    "learning_rate": params["learning_rate"],
                    "weight_decay": params["weight_decay"],
                    "batch_size": params["batch_size"],
                    **metrics,
                }
            )

    with open("./metrics/all_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv("./metrics/all_results.csv", index=False)

if __name__ == "__main__":
    main()
