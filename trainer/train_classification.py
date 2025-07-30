import sys
import os
import json
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, ".."))
sys.path.insert(0, parent_dir)

from preprocessing.preprocess_ade import get_tokenizer
from model_selection.models_factory import get_model
from preprocessing.tokenizer_utils import tokenize_for_model

def compute_metrics(eval_pred, target_names):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    labels = labels.astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

    report = classification_report(labels, preds, target_names=target_names, output_dict=True)
    for label in target_names:
        metrics[f"{label}_support"] = report[label]["support"]
        metrics[f"{label}_precision"] = report[label]["precision"]
        metrics[f"{label}_recall"] = report[label]["recall"]
        metrics[f"{label}_f1-score"] = report[label]["f1-score"]

    return metrics

def main(dataset_name="psytar_classification"):

    dataset_map = {
        "ade_classification": {
            "path": os.path.join(base_dir, "..", "data_sets", "ade_corpus_dataset", "ade_corpus_classification.csv"),
            "target_names": ["not-related", "related"],
        },
        "psytar_classification": {
            "path": os.path.join(base_dir, "..", "data_sets", "psytar_dataset", "psytar_classification.csv"),
            "target_names": ["not-related", "related"],
        },
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Dataset {dataset_name} non supportato. Scegli tra {list(dataset_map.keys())}")

    data_info = dataset_map[dataset_name]
    df = pd.read_csv(data_info["path"])
    df = df.sample(n=300, random_state=42).reset_index(drop=True)

    texts = df["text"].tolist()
    labels = df["label"].tolist()
    target_names = data_info["target_names"]

    tokenizer = get_tokenizer(model_type="classification")

    param_grid = [
        {"learning_rate": 5e-5, "weight_decay": 0.01, "batch_size": 8},
        {"learning_rate": 3e-5, "weight_decay": 0.01, "batch_size": 16},
        {"learning_rate": 2e-5, "weight_decay": 0.0, "batch_size": 8},
    ]

    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    all_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(texts, labels), start=1):
        print(f"\n--- Fold {fold} ---")

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

        train_dataset = train_dataset.map(lambda x: tokenize_for_model(x, tokenizer), batched=True)
        val_dataset = val_dataset.map(lambda x: tokenize_for_model(x, tokenizer), batched=True)

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        for i, params in enumerate(param_grid, start=1):
            print(f"Fold {fold}, Set {i} - Params: {params}")

            model = get_model(num_labels=len(target_names), model_type="classification")

            training_args = TrainingArguments(
                output_dir=f"./results/fold_{fold}/set_{i}",
                logging_dir=f"./logs/fold_{fold}/set_{i}",
                per_device_train_batch_size=params["batch_size"],
                per_device_eval_batch_size=params["batch_size"],
                num_train_epochs=1,
                learning_rate=params["learning_rate"],
                weight_decay=params["weight_decay"],
                save_strategy="epoch",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=lambda p: compute_metrics(p, target_names),
            )

            trainer.train()
            metrics = trainer.evaluate()

            os.makedirs("./metrics", exist_ok=True)
            with open(f"./metrics/fold_{fold}_set_{i}.json", "w") as f:
                json.dump(metrics, f)

            os.makedirs("./models", exist_ok=True)
            model.save_pretrained(f"./models/bert_model_fold_{fold}_set_{i}")
            tokenizer.save_pretrained(f"./models/bert_model_fold_{fold}_set_{i}")

            all_results.append(
                {
                    "fold": fold,
                    "param_set": i,
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
    print(">>> Inizio test del training su 'ade_classification'...")

    try:
        main("ade_classification")
        print(">>> Training completato correttamente ✅")
    except Exception as e:
        print(">>> Errore durante il training ❌")
        print(e)

