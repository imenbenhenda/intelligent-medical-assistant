# src/evaluate.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
import json
from tqdm import tqdm
import os

# -------------------------
# Config
# -------------------------
model_name_or_path = "models/biobert-classifier"  # Vérifiez que ce dossier existe
test_file = "./data/processed/test_clean.jsonl"   # Vérifiez le bon fichier
batch_size = 16
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"📍 Device: {device}")
print(f"📍 Test file: {test_file}")
print(f"📍 Model path: {model_name_or_path}")

# -------------------------
# Vérifications initiales
# -------------------------
if not os.path.exists(model_name_or_path):
    print(f"❌ ERREUR: Modèle introuvable à {model_name_or_path}")
    exit(1)

if not os.path.exists(test_file):
    print(f"❌ ERREUR: Fichier de test introuvable: {test_file}")
    # Liste les fichiers disponibles
    processed_dir = "./data/processed"
    if os.path.exists(processed_dir):
        print("📁 Fichiers disponibles dans data/processed:")
        for f in os.listdir(processed_dir):
            print(f"   - {f}")
    exit(1)

# -------------------------
# Charger modèle et tokenizer
# -------------------------
print("🔄 Chargement du modèle...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.eval()
    model.to(device)
    print("✅ Modèle chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    exit(1)

# -------------------------
# Charger données de test
# -------------------------
print("🔄 Chargement des données de test...")
texts, true_labels = [], []

try:
    with open(test_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Lecture du fichier"):
            item = json.loads(line.strip())
            # Vérifier la structure du fichier
            if "sections" in item:
                # Format original avec sections
                for section_name, section_text in item["sections"].items():
                    if section_text and section_text.strip():
                        texts.append(section_text)
                        true_labels.append(section_name)
            elif "text" in item and "label" in item:
                # Format déjà flatten
                texts.append(item["text"])
                true_labels.append(item["label"])
            else:
                print(f"⚠️ Format inattendu: {item.keys()}")
                
except Exception as e:
    print(f"❌ Erreur lors du chargement des données: {e}")
    exit(1)

print(f"📊 {len(texts)} exemples chargés")

if len(texts) == 0:
    print("❌ Aucune donnée chargée!")
    exit(1)

# -------------------------
# Mapping des labels (identique à l'entraînement)
# -------------------------
LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

# Convertir les labels textuels en IDs
try:
    true_label_ids = [label2id[label] for label in true_labels]
except KeyError as e:
    print(f"❌ Label inconnu trouvé: {e}")
    print("Labels uniques dans les données:", set(true_labels))
    exit(1)

# -------------------------
# Fonction de prédiction par batch (BEAUCOUP plus rapide)
# -------------------------
def predict_batch(texts, batch_size=16):
    predictions = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Prédictions"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenizer le batch
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        
        # Déplacer sur le device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Prédire
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Récupérer les prédictions
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(batch_preds)
    
    return predictions

# -------------------------
# Faire les prédictions
# -------------------------
print("🎯 Début des prédictions...")
predicted_label_ids = predict_batch(texts, batch_size)

# Convertir les IDs en labels textuels
predicted_labels = [id2label[pred_id] for pred_id in predicted_label_ids]

# -------------------------
# Calculer métriques
# -------------------------
print("\n" + "="*50)
print("📊 RÉSULTATS D'ÉVALUATION")
print("="*50)

accuracy = accuracy_score(true_label_ids, predicted_label_ids)
f1 = f1_score(true_label_ids, predicted_label_ids, average="weighted")
recall = recall_score(true_label_ids, predicted_label_ids, average="weighted")
precision = precision_score(true_label_ids, predicted_label_ids, average="weighted")

print(f"✅ Accuracy  : {accuracy:.4f}")
print(f"✅ F1-score  : {f1:.4f}")
print(f"✅ Recall    : {recall:.4f}")
print(f"✅ Precision : {precision:.4f}")

# -------------------------
# Rapport détaillé
# -------------------------
print("\n" + "="*50)
print("📈 RAPPORT DÉTAILLÉ PAR CLASSE")
print("="*50)
print(classification_report(true_label_ids, predicted_label_ids, 
                           target_names=LABELS, digits=4))

# -------------------------
# Exemples de prédictions
# -------------------------
print("\n" + "="*50)
print("🔍 EXEMPLES DE PRÉDICTIONS")
print("="*50)

for i in range(min(5, len(texts))):
    print(f"\nExemple {i+1}:")
    print(f"Texte: {texts[i][:100]}...")
    print(f"Vrai label: {true_labels[i]}")
    print(f"Prédiction: {predicted_labels[i]}")
    print(f"✅ Correct" if true_labels[i] == predicted_labels[i] else "❌ Incorrect")