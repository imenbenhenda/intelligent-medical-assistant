# src/evaluate.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
import json
from tqdm import tqdm
import os

# -------------------------
# 1 Configuration
# -------------------------
model_name_or_path = "models/biobert-classifier"
test_file = "./data/processed/test_clean.jsonl"
batch_size = 16
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"Test file: {test_file}")
print(f"Model path: {model_name_or_path}")

# -------------------------
# 2 Initial checks
# -------------------------
if not os.path.exists(model_name_or_path):
    print(f"ERROR: Model not found at {model_name_or_path}")
    exit(1)

if not os.path.exists(test_file):
    print(f"ERROR: Test file not found: {test_file}")
    processed_dir = "./data/processed"
    if os.path.exists(processed_dir):
        print("Available files in data/processed:")
        for f in os.listdir(processed_dir):
            print(f"   - {f}")
    exit(1)

# -------------------------
# 3 Load model and tokenizer
# -------------------------
print("Loading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.eval()
    model.to(device)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# -------------------------
# 4 Load test data
# -------------------------
print("Loading test data...")
texts, true_labels = [], []

try:
    with open(test_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading file"):
            item = json.loads(line.strip())
            # Check file structure
            if "sections" in item:
                # Original format with sections
                for section_name, section_text in item["sections"].items():
                    if section_text and section_text.strip():
                        texts.append(section_text)
                        true_labels.append(section_name)
            elif "text" in item and "label" in item:
                # Already flattened format
                texts.append(item["text"])
                true_labels.append(item["label"])
            else:
                print(f"Unexpected format: {item.keys()}")
                
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

print(f"Loaded {len(texts)} examples")

if len(texts) == 0:
    print("No data loaded!")
    exit(1)

# -------------------------
# 5 Label mapping (same as training)
# -------------------------
LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

# Convert text labels to IDs
try:
    true_label_ids = [label2id[label] for label in true_labels]
except KeyError as e:
    print(f"Unknown label found: {e}")
    print("Unique labels in data:", set(true_labels))
    exit(1)

# -------------------------
# 6 Batch prediction function (much faster)
# -------------------------
def predict_batch(texts, batch_size=16):
    predictions = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Predictions"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predictions
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(batch_preds)
    
    return predictions

# -------------------------
# 7 Make predictions
# -------------------------
print("Starting predictions...")
predicted_label_ids = predict_batch(texts, batch_size)

# Convert IDs to text labels
predicted_labels = [id2label[pred_id] for pred_id in predicted_label_ids]

# -------------------------
# 8 Calculate metrics
# -------------------------
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)

accuracy = accuracy_score(true_label_ids, predicted_label_ids)
f1 = f1_score(true_label_ids, predicted_label_ids, average="weighted")
recall = recall_score(true_label_ids, predicted_label_ids, average="weighted")
precision = precision_score(true_label_ids, predicted_label_ids, average="weighted")

print(f"Accuracy  : {accuracy:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")

# -------------------------
# 9 Detailed report
# -------------------------
print("\n" + "="*50)
print("DETAILED CLASS REPORT")
print("="*50)
print(classification_report(true_label_ids, predicted_label_ids, 
                           target_names=LABELS, digits=4))

# -------------------------
# 10 Prediction examples
# -------------------------
print("\n" + "="*50)
print("PREDICTION EXAMPLES")
print("="*50)

for i in range(min(5, len(texts))):
    print(f"\nExample {i+1}:")
    print(f"Text: {texts[i][:100]}...")
    print(f"True label: {true_labels[i]}")
    print(f"Prediction: {predicted_labels[i]}")
    print(f"Correct" if true_labels[i] == predicted_labels[i] else "Incorrect")