import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import KFold
from model import get_model, get_tokenizer, tokenize_for_classification  # Importa anche la funzione tokenize

def main():
    df = pd.read_csv("data/binary/full.csv")
    texts = df["sentences"].tolist()
    labels =df[["ADR", "WD", "EF", "INF", "SSI", "DI"]].values.tolist()

    tokenizer = get_tokenizer()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
        print(f"\n--- Fold {fold + 1} ---")

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

        
        train_dataset = train_dataset.map(lambda x: tokenize_for_classification(x, tokenizer), batched=True)
        val_dataset = val_dataset.map(lambda x: tokenize_for_classification(x, tokenizer), batched=True)

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        model = get_model(num_labels=2)

        training_args = TrainingArguments(
            output_dir=f"./results/fold_{fold + 1}",
            evaluation_strategy="epoch",
            logging_dir=f"./logs/fold_{fold + 1}",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(-1)
            accuracy = (preds == labels).astype(float).mean()
            return {"accuracy": accuracy}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        model.save_pretrained(f"./bert_model_fold_{fold + 1}")
        tokenizer.save_pretrained(f"./bert_model_fold_{fold + 1}")

if __name__ == "__main__":
    main()
