import sys
import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import Trainer
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# Import preprocessing and model loading functions
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, ".."))
sys.path.insert(0, parent_dir)

from preprocessing.preprocess_ade import get_tokenizer, preprocess_ade_csv
from preprocessing.preprocess_psytar import preprocess_psytar_conll
from model_selection.models_factory import get_model

# Label list
label_list = ["O", "B-DRUG", "I-DRUG", "B-EFFECT", "I-EFFECT"]

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

def main(dataset_name="ade_ner", models_dir="./models", metrics_dir="./metrics_eval"):
    os.makedirs(metrics_dir, exist_ok=True)

    dataset_map = {
        "ade_ner": {
            "path": os.path.join(base_dir, "..", "data_sets", "ade_corpus_dataset", "ade_corpus_ner.csv"),
            "preprocess_fn": preprocess_ade_csv,
            "label_list": label_list
        },
        "psytar_ner": {
            "path": os.path.join(base_dir, "..", "data_sets", "psytar_dataset", "psytar_ner.txt"),
            "preprocess_fn": preprocess_psytar_conll,
            "label_list": label_list
        }
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from {list(dataset_map.keys())}")

    # Load dataset
    data_info = dataset_map[dataset_name]
    tokenizer = get_tokenizer(model_type="ner")
    dataset_encodings = data_info["preprocess_fn"](data_info["path"], tokenizer, max_len=128)
    dataset = Dataset.from_list(dataset_encodings)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    all_eval_results = []

    # Iterate over all saved models
    for model_dir in sorted(os.listdir(models_dir)):
        model_path = os.path.join(models_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        print(f"\nEvaluating model: {model_dir}")
        model = get_model(num_labels=len(label_list), model_type="ner")
        model.from_pretrained(model_path)
        tokenizer = tokenizer.from_pretrained(model_path)

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        metrics = trainer.evaluate(eval_dataset=dataset)

        # Save metrics
        with open(os.path.join(metrics_dir, f"{model_dir}_eval.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        metrics["model"] = model_dir
        all_eval_results.append(metrics)

    # Save summary CSV
    df_results = pd.DataFrame(all_eval_results)
    df_results.to_csv(os.path.join(metrics_dir, "all_eval_results.csv"), index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ade_ner", help="Dataset name: ade_ner or psytar_ner")
    parser.add_argument("--models_dir", type=str, default="./models", help="Directory with trained models")
    parser.add_argument("--metrics_dir", type=str, default="./metrics_eval", help="Directory to save evaluation metrics")
    args = parser.parse_args()

    main(dataset_name=args.dataset, models_dir=args.models_dir, metrics_dir=args.metrics_dir)
