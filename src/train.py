"""
train.py - Fine-tuning BioBERT pour la classification des phrases
Version test rapide pour prototype
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
# CONFIGURATION GLOBALE
# -------------------------------
MODEL_NAME_CLS = "dmis-lab/biobert-base-cased-v1.1"
OUTPUT_DIR_CLS = "models/biobert-classifier"
LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# 1. CHARGEMENT DES DONNÉES
# -------------------------------
logger.info("Chargement des données prétraitées...")

dataset = load_dataset("json", data_files={
    "train": "data/processed/train_clean.jsonl",
    "validation": "data/processed/dev_clean.jsonl"
})

print("Colonnes initiales :", dataset["train"].column_names)
print("Premier article :", dataset["train"][0])

# -------- FLATTEN DES SECTIONS --------
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

print("Nouvelles colonnes :", dataset["train"].column_names)
print("Premier exemple après flatten :", dataset["train"][0])

# -------- LIMITER LE DATASET POUR TEST RAPIDE --------
dataset["train"] = dataset["train"].select(range(min(20000, len(dataset["train"]))))
dataset["validation"] = dataset["validation"].select(range(min(2000, len(dataset["validation"]))))
logger.info(f"Dataset limité : {len(dataset['train'])} exemples pour entraînement, {len(dataset['validation'])} pour validation")

# -------------------------------
# 2. PRÉPARATION DES LABELS
# -------------------------------
logger.info("Préparation des labels...")

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

print("Mapping des labels :", label2id)

def convert_labels_to_ids(example):
    example['label'] = label2id[example['label']]
    return example

dataset = dataset.map(convert_labels_to_ids)

# -------------------------------
# 3. TOKENIZER + ENCODAGE
# -------------------------------
logger.info("Tokenisation des phrases...")

tokenizer_cls = AutoTokenizer.from_pretrained(MODEL_NAME_CLS)

def tokenize_function(batch):
    encodings = tokenizer_cls(
        batch["text"],
        truncation=True,
        padding=True,
        max_length=64  # Séquence plus courte pour accélérer
    )
    encodings["labels"] = batch["label"]
    return encodings

dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000
)

# -------------------------------
# 4. MODELE DE CLASSIFICATION
# -------------------------------
logger.info("Initialisation du modèle BioBERT...")

model_cls = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_CLS,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)

# -------------------------------
# 5. METRIQUES
# -------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# -------------------------------
# 6. ARGUMENTS D'ENTRAINEMENT
# -------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR_CLS,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,  # Une seule époque pour test rapide
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
# 9. ENTRAINEMENT
# -------------------------------
logger.info("Démarrage de l'entraînement BioBERT...")
logger.info(f"Taille du dataset d'entraînement: {len(dataset['train'])}")
logger.info(f"Taille du dataset de validation: {len(dataset['validation'])}")

trainer.train()
trainer.save_model(OUTPUT_DIR_CLS)
tokenizer_cls.save_pretrained(OUTPUT_DIR_CLS)

logger.info("Modèle de classification entraîné et sauvegardé ✅")
