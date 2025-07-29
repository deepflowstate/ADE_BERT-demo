import pandas as pd
import json
import os
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from BERTmodel import get_model, get_tokenizer, tokenize_for_classification


def main(data_path=None):
    if data_path is None:
        base_dir = os.path.dirname(__file__)
        data_path = os.path.join(
            base_dir, "..", "data_sets", "ade_corpus_dataset", "ade_corpus_classification.csv"
        )

    # load the dataset
    df = pd.read_csv(data_path)
    # Just for testing we train the model just on 1000 examples and not the whole dataset
    df = df.sample(n=1000, random_state=42).reset_index(drop=True)

    param_grid = [
        {"learning_rate": 5e-5, "weight_decay": 0.01, "batch_size": 8},
        {"learning_rate": 3e-5, "weight_decay": 0.01, "batch_size": 16},
        {"learning_rate": 2e-5, "weight_decay": 0.0, "batch_size": 8},
    ]

    target_names = ["not-related", "related"]

    all_results = []

    # Converting the two columns
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # Initializing tokenizer
    tokenizer = get_tokenizer()
    # Stratified 5-fold cross-validation
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    stratify_labels = [lb for lb in labels]

    for fold, (train_idx, val_idx) in enumerate(kf.split(texts, stratify_labels)):
        print(f"\n--- Fold {fold + 1} ---")

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # Using Hugging Face
        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

        train_dataset = train_dataset.map(
            lambda x: tokenize_for_classification(x, tokenizer), batched=True
        )
        val_dataset = val_dataset.map(
            lambda x: tokenize_for_classification(x, tokenizer), batched=True
        )

        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        val_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        for i, params in enumerate(param_grid):
            print(f"Fold {fold + 1}, Set {i + 1} - Params: {params}")

            model = get_model(num_labels=2, model_type="classification")

            training_args = TrainingArguments(
                output_dir=f"./results/fold_{fold + 1}/set_{i + 1}",
                logging_dir=f"./logs/fold_{fold + 1}/set_{i + 1}",
                per_device_train_batch_size=params["batch_size"],
                per_device_eval_batch_size=params["batch_size"],
                num_train_epochs=1,
                learning_rate=params["learning_rate"],
                weight_decay=params["weight_decay"],
                save_strategy="epoch",
            )

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                preds = logits.argmax(axis=1)
                labels = labels.astype(int)

                metrics = {
                    "accuracy": accuracy_score(labels, preds),
                    "precision_micro": precision_score(
                        labels, preds, average="macro"
                    ),  # macro for unbalanced classes
                    "recall_micro": recall_score(labels, preds, average="macro"),
                    "f1_micro": f1_score(labels, preds, average="macro"),
                }

                report = classification_report(
                    labels, preds, target_names=target_names, output_dict=True
                )

                for label in target_names:
                    metrics[f"{label}_support"] = report[label]["support"]
                    metrics[f"{label}_precision"] = report[label]["precision"]
                    metrics[f"{label}_recall"] = report[label]["recall"]
                    metrics[f"{label}_f1-score"] = report[label]["f1-score"]

                return metrics

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
            )

            trainer.train()
            print(f"fold {fold + 1} completed")
            metrics = trainer.evaluate()

            os.makedirs("./metrics", exist_ok=True)
            with open(f"./metrics/fold_{fold + 1}_set_{i + 1}.json", "w") as f:
                json.dump(metrics, f)

            os.makedirs("./models", exist_ok=True)
            model.save_pretrained(f"./models/bert_model_fold_{fold + 1}_set_{i + 1}")
            tokenizer.save_pretrained(
                f"./models/bert_model_fold_{fold + 1}_set_{i + 1}"
            )

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
