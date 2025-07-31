import os
import sys
import json
import pandas as pd
from datasets import Dataset
from transformers import Trainer
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# Add parent directory to path
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, ".."))
sys.path.insert(0, parent_dir)

from preprocessing.preprocess_ade import get_tokenizer
from model_selection.models_factory import get_model
from preprocessing.tokenizer_utils import tokenize_for_model

def compute_metrics(eval_pred, target_names):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

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

def evaluate_model(model_dir, dataset_path, target_names):
    """Evaluate a single saved model on a dataset"""
    print(f"Evaluating {model_dir} on {os.path.basename(dataset_path)}")

    # Load model and tokenizer
    tokenizer = get_tokenizer(model_type="classification").from_pretrained(model_dir)
    model = get_model(num_labels=len(target_names), model_type="classification").from_pretrained(model_dir)

    # Load dataset
    df = pd.read_csv(dataset_path)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    eval_dataset = Dataset.from_dict({"text": texts, "label": labels})
    eval_dataset = eval_dataset.map(lambda x: tokenize_for_model(x, tokenizer), batched=True)
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Evaluate
    trainer = Trainer(model=model)
    preds_output = trainer.predict(eval_dataset)
    metrics = compute_metrics((preds_output.predictions, preds_output.label_ids), target_names)
    
    return metrics

if __name__ == "__main__":
    # Define datasets
    dataset_map = {
        "ade_classification": {
            "path": os.path.join(parent_dir, "data_sets", "ade_corpus_dataset", "ade_corpus_classification.csv"),
            "target_names": ["not-related", "related"],
        },
        "psytar_classification": {
            "path": os.path.join(parent_dir, "data_sets", "psytar_dataset", "psytar_classification.csv"),
            "target_names": ["not-related", "related"],
        },
    }

    model_root = "./models"
    results = []

    for model_dir in sorted(os.listdir(model_root)):
        full_model_path = os.path.join(model_root, model_dir)
        if not os.path.isdir(full_model_path):
            continue

        for dataset_name, info in dataset_map.items():
            metrics = evaluate_model(full_model_path, info["path"], info["target_names"])
            metrics.update({
                "model": model_dir,
                "dataset": dataset_name,
            })
            results.append(metrics)

            # Save per-model-per-dataset metrics
            os.makedirs("./eval_metrics", exist_ok=True)
            with open(f"./eval_metrics/{model_dir}_{dataset_name}_eval.json", "w") as f:
                json.dump(metrics, f, indent=4)

    # Save combined results
    df_results = pd.DataFrame(results)
    df_results.to_csv("./eval_metrics/all_eval_results.csv", index=False)
    print("Evaluation completed. Combined results saved to ./eval_metrics/all_eval_results.csv")
