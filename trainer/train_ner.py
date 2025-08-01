import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, ".."))
sys.path.insert(0, parent_dir)

from preprocessing.preprocess_ade import get_tokenizer, preprocess_ade_csv
from preprocessing.preprocess_psytar import preprocess_psytar_conll
from model_selection.models_factory import get_model

import json
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import KFold
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# List of NER labels used for evaluation and mapping
label_list = ["O", "B-DRUG", "I-DRUG", "B-EFFECT", "I-EFFECT"]

def compute_metrics(p):
    """
    Compute evaluation metrics for NER using seqeval.

    Args:
        p (tuple): (predictions, labels)
            - predictions: model output logits
            - labels: ground-truth label IDs (with -100 for padding)

    Returns:
        dict: macro-averaged precision, recall, F1, plus per-entity scores
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)  # Convert logits to predicted class IDs

    # Remove masked labels (-100) and convert IDs to string labels
    true_labels = [
        [label_list[l] for l in label if l != -100] for label in labels
    ]
    true_preds = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute and return classification metrics
    results = {
        "precision": precision_score(true_labels, true_preds, average="macro"),
        "recall": recall_score(true_labels, true_preds, average="macro"),
        "f1": f1_score(true_labels, true_preds, average="macro"),
    }
    results.update(classification_report(true_labels, true_preds, output_dict=True))
    return results

def main(dataset_name="ade_ner"):
    """
    Main training loop for NER with k-fold cross-validation.

    Args:
        dataset_name (str): one of ["ade_ner", "psytar_ner"]

    Workflow:
        - Load the dataset and preprocessing function
        - Tokenize the dataset
        - Apply 3-fold cross-validation
        - Train and evaluate model with multiple hyperparameter sets
        - Save model checkpoints and metrics
    """
    base_dir = os.path.dirname(__file__)

    # Dataset configuration mapping
    dataset_map = {
        "ade_ner": {
            "path": os.path.join(base_dir, "..", "data_sets", "ade_corpus_dataset", "ade_corpus_ner.csv"),
            "preprocess_fn": preprocess_ade_csv,
            "label_list": ["O", "B-DRUG", "I-DRUG", "B-EFFECT", "I-EFFECT"]
        },
        "psytar_ner": {
            "path": os.path.join(base_dir, "..", "data_sets", "psytar_dataset", "psytar_ner.txt"),
            "preprocess_fn": preprocess_psytar_conll,
            "label_list": ["O", "B-DRUG", "I-DRUG", "B-EFFECT", "I-EFFECT"]  
        }
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from {list(dataset_map.keys())}")

    # Load preprocessing config
    data_info = dataset_map[dataset_name]
    data_path = data_info["path"]
    preprocess_fn = data_info["preprocess_fn"]
    label_list = data_info["label_list"]
    max_len = 128

    # Load tokenizer
    tokenizer = get_tokenizer(model_type="ner")

    # Preprocess and tokenize dataset
    dataset_encodings = preprocess_fn(data_path, tokenizer, max_len=max_len)
    dataset = Dataset.from_list(dataset_encodings)

    # Hyperparameter grid for model selection
    param_grid = [
        {"learning_rate": 5e-5, "weight_decay": 0.01, "batch_size": 8},
        {"learning_rate": 3e-5, "weight_decay": 0.01, "batch_size": 16},
        {"learning_rate": 2e-5, "weight_decay": 0.0, "batch_size": 8},
    ]
    
    # 3-fold cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=49)
    all_results = []

    # Iterate over folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n--- Fold {fold + 1} ---")

        # Split data
        train_dataset = dataset.select(train_idx)
        val_dataset = dataset.select(val_idx)

        # Prepare data for Trainer
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Try each hyperparameter configuration
        for i, params in enumerate(param_grid):
            print(f"Fold {fold + 1}, Set {i + 1} - Params: {params}")

            # Load model
            model = get_model(num_labels=len(label_list), model_type="ner")

            # Define training arguments
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

            # Set up Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
            )

            # Train and evaluate
            trainer.train()
            metrics = trainer.evaluate()

            # Save metrics
            os.makedirs("./metrics", exist_ok=True)
            with open(f"./metrics/fold_{fold + 1}_set_{i + 1}.json", "w") as f:
                json.dump(metrics, f)

            # Save model and tokenizer
            os.makedirs("./models", exist_ok=True)
            model.save_pretrained(f"./models/fold_{fold + 1}_set_{i + 1}")
            tokenizer.save_pretrained(f"./models/fold_{fold + 1}_set_{i + 1}")

            # Store results
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

    # Save all results to file
    with open("./metrics/all_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    # Save also as CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("./metrics/all_results.csv", index=False)

if __name__ == "__main__":
    main()
