"""
train.py - Fine-tuning BioBERT for sentence classification
Quick test version for prototype
"""

import os
import logging
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# -------------------------------
# GLOBAL CONFIGURATION
# -------------------------------
MODEL_NAME_CLS = "dmis-lab/biobert-base-cased-v1.1"
OUTPUT_DIR_CLS = "models/biobert-classifier"
LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# 1. DATA LOADING
# -------------------------------
logger.info("Loading preprocessed data...")

dataset = load_dataset("json", data_files={
    "train": "data/processed/train_clean.jsonl",
    "validation": "data/processed/dev_clean.jsonl"
})

print("Initial columns:", dataset["train"].column_names)
print("First article:", dataset["train"][0])

# -------- FLATTEN SECTIONS --------
def flatten_dataset(dataset_split):
    texts, labels = [], []
    for article in dataset_split:
        for section_name, section_text in article["sections"].items():
            if section_text is None or section_text.strip() == "":
                continue
            texts.append(section_text)
            labels.append(section_name)
    return Dataset.from_dict({"text": texts, "label": labels})

dataset = DatasetDict({
    "train": flatten_dataset(dataset["train"]),
    "validation": flatten_dataset(dataset["validation"])
})

print("New columns:", dataset["train"].column_names)
print("First example after flatten:", dataset["train"][0])

# -------- LIMIT DATASET FOR QUICK TEST --------
dataset["train"] = dataset["train"].select(range(min(20000, len(dataset["train"]))))
dataset["validation"] = dataset["validation"].select(range(min(2000, len(dataset["validation"]))))
logger.info(f"Limited dataset: {len(dataset['train'])} training examples, {len(dataset['validation'])} validation examples")

# -------------------------------
# 2. LABEL PREPARATION
# -------------------------------
logger.info("Preparing labels...")

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

print("Label mapping:", label2id)

def convert_labels_to_ids(example):
    example['label'] = label2id[example['label']]
    return example

dataset = dataset.map(convert_labels_to_ids)

# -------------------------------
# 3. TOKENIZER + ENCODING
# -------------------------------
logger.info("Tokenizing sentences...")

tokenizer_cls = AutoTokenizer.from_pretrained(MODEL_NAME_CLS)

def tokenize_function(batch):
    encodings = tokenizer_cls(
        batch["text"],
        truncation=True,
        padding=True,
        max_length=64  # Shorter sequence for speed
    )
    encodings["labels"] = batch["label"]
    return encodings

dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000
)

# -------------------------------
# 4. CLASSIFICATION MODEL
# -------------------------------
logger.info("Initializing BioBERT model...")

model_cls = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_CLS,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)

# -------------------------------
# 5. METRICS
# -------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# -------------------------------
# 6. TRAINING ARGUMENTS
# -------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR_CLS,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,  # One epoch for quick test
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    save_total_limit=2,
    dataloader_pin_memory=False
)

# -------------------------------
# 7. DATA COLLATOR
# -------------------------------
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer_cls,
    padding=True
)

# -------------------------------
# 8. TRAINER
# -------------------------------
trainer = Trainer(
    model=model_cls,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# -------------------------------
# 9. TRAINING
# -------------------------------
logger.info("Starting BioBERT training...")
logger.info(f"Training dataset size: {len(dataset['train'])}")
logger.info(f"Validation dataset size: {len(dataset['validation'])}")

trainer.train()
trainer.save_model(OUTPUT_DIR_CLS)
tokenizer_cls.save_pretrained(OUTPUT_DIR_CLS)

logger.info("Classification model trained and saved ✅")