# src/model.py
"""
LLM-based pipeline module for MedicalLiteratureAssistant
- Sentence classification with BioBERT
- Article summarization with T5
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -----------------------------
# 1 Sentence Classification
# -----------------------------
# Labels for each sentence type
LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]

# Load BioBERT tokenizer and model
tokenizer_cls = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model_cls = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    num_labels=len(LABELS)
)
model_cls.eval()  # evaluation mode

def classify_sentence(sentence, model=model_cls, tokenizer=tokenizer_cls):
    """
    Predicts the label of a medical sentence
    Args:
        sentence (str): sentence text
    Returns:
        str: predicted label
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred_id = torch.argmax(logits, dim=1).item()
    return LABELS[pred_id]

# -----------------------------
# 2 Article Summarization
# -----------------------------
# Load T5 tokenizer and model
tokenizer_summ = T5Tokenizer.from_pretrained("t5-small")
model_summ = T5ForConditionalGeneration.from_pretrained("t5-small")
model_summ.eval()

def summarize_article(article_sentences, model=model_summ, tokenizer=tokenizer_summ, max_len=150):
    """
    Generates a summary from a list of classified sentences
    Args:
        article_sentences (list[str]): classified sentences with their labels
    Returns:
        str: generated summary
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
# 3 Complete Pipeline: classification + summarization
# -----------------------------
def process_article(article_sentences):
    """
    Complete pipeline:
    - Classifies each sentence
    - Generates a structured summary
    Args:
        article_sentences (list[dict]): [{"label":"UNKNOWN","text":"..."}, ...]
    Returns:
        str: article summary
    """
    classified_sentences = []
    for s in article_sentences:
        text = s['text']
        label = classify_sentence(text)
        classified_sentences.append(f"{label}: {text}")

    summary = summarize_article(classified_sentences)
    return summary

# -----------------------------
# 4 Quick usage example
# -----------------------------
if __name__ == "__main__":
    # Example article
    example_article = [
        {"label": "UNKNOWN", "text": "This study investigates the effect of drug X on patients."},
        {"label": "UNKNOWN", "text": "We performed a double-blind randomized trial."},
        {"label": "UNKNOWN", "text": "The results showed significant improvement in symptoms."},
        {"label": "UNKNOWN", "text": "Conclusion: Drug X is effective and safe for patients."}
    ]

    summary = process_article(example_article)
    print("Generated summary:\n", summary)