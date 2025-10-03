# src/model.py
"""
Module pour le pipeline LLM-based de MedicalLiteratureAssistant
- Classification de phrases avec BioBERT
- Résumé d'article avec T5
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -----------------------------
# 1️⃣ Classification des phrases
# -----------------------------
# Labels pour chaque type de phrase
LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]

# Charger le tokenizer et le modèle BioBERT
tokenizer_cls = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model_cls = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    num_labels=len(LABELS)
)
model_cls.eval()  # mode évaluation

def classify_sentence(sentence, model=model_cls, tokenizer=tokenizer_cls):
    """
    Prédit le label d'une phrase médicale
    Args:
        sentence (str): texte de la phrase
    Returns:
        str: label prédit
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred_id = torch.argmax(logits, dim=1).item()
    return LABELS[pred_id]

# -----------------------------
# 2️⃣ Résumé des articles
# -----------------------------
# Charger le tokenizer et le modèle T5
tokenizer_summ = T5Tokenizer.from_pretrained("t5-small")
model_summ = T5ForConditionalGeneration.from_pretrained("t5-small")
model_summ.eval()

def summarize_article(article_sentences, model=model_summ, tokenizer=tokenizer_summ, max_len=150):
    """
    Génère un résumé à partir d'une liste de phrases classées
    Args:
        article_sentences (list[str]): phrases classées avec leur label
    Returns:
        str: résumé généré
    """
    input_text = "summarize: " + " ".join(article_sentences)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        **inputs,
        max_length=max_len,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# -----------------------------
# 3️⃣ Pipeline complet : classification + résumé
# -----------------------------
def process_article(article_sentences):
    """
    Pipeline complet :
    - Classifie chaque phrase
    - Génère un résumé structuré
    Args:
        article_sentences (list[dict]): [{"label":"UNKNOWN","text":"..."}, ...]
    Returns:
        str: résumé de l'article
    """
    classified_sentences = []
    for s in article_sentences:
        text = s['text']
        label = classify_sentence(text)
        classified_sentences.append(f"{label}: {text}")

    summary = summarize_article(classified_sentences)
    return summary

# -----------------------------
# 4️⃣ Exemple rapide d'utilisation
# -----------------------------
if __name__ == "__main__":
    # Exemple d'article
    example_article = [
        {"label": "UNKNOWN", "text": "This study investigates the effect of drug X on patients."},
        {"label": "UNKNOWN", "text": "We performed a double-blind randomized trial."},
        {"label": "UNKNOWN", "text": "The results showed significant improvement in symptoms."},
        {"label": "UNKNOWN", "text": "Conclusion: Drug X is effective and safe for patients."}
    ]

    summary = process_article(example_article)
    print("Résumé généré :\n", summary)
