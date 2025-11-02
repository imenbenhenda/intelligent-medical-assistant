# src/preprocessing.py

import re
import json
from tqdm import tqdm
from src.utils import iter_pubmed, ensure_dir

# =========================
# Text cleaning function
# =========================
def clean_text(text, lower=True, remove_control_chars=True, keep_unicode_letters=True):
    """
    Cleans sentence text.
    
    Parameters:
    - text: sentence to clean
    - lower: convert to lowercase if True
    - remove_control_chars: replace line breaks/tabs with spaces
    - keep_unicode_letters: keep accented letters (useful for medical vocabulary)
    
    Returns:
    - cleaned text ready for model
    """
    if text is None:
        return ""

    # 1) Convert to lowercase
    if lower:
        text = text.lower()

    # 2) Remove line breaks and tabs
    if remove_control_chars:
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # 3) Remove unwanted characters
    if keep_unicode_letters:
        # Keep accented letters, numbers, basic punctuation
        text = re.sub(r"[^0-9A-Za-z\u00C0-\u024F\.\,\:\;\(\)\[\]\%\+\-\/\?\!\s]", " ", text)
    else:
        # Without accents
        text = re.sub(r"[^0-9A-Za-z\.\,\:\;\(\)\[\]\%\+\-\/\?\!\s]", " ", text)

    # 4) Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================
# Number anonymization
# =========================
def replace_numbers_with_at(text):
    """
    Replaces all numbers with '@'
    - useful for anonymizing digits and numerical values
    """
    return re.sub(r"\d+([.,]\d+)?", "@", text)

# =========================
# Function to group sentences by sections
# =========================
def group_sections(article_sentences, clean_fn=clean_text, replace_numbers=False):
    """
    Transforms an article into a dictionary of sections.
    
    Parameters:
    - article_sentences: list of dicts {'label','text'} for an article
    - clean_fn: function to clean each sentence
    - replace_numbers: if True, replaces all numbers with '@'

    Returns:
    - sections_joined: dict {label: "joined complete text"}
    """
    sections = {}

    for s in article_sentences:
        label = s.get("label", "UNKNOWN")
        text = s.get("text", "")
        
        # Clean text
        cleaned = clean_fn(text)
        
       
        if replace_numbers:
            cleaned = replace_numbers_with_at(cleaned)
        
        # Add cleaned sentence to corresponding section
        sections.setdefault(label, []).append(cleaned)

    # Join all sentences from same section into paragraph
    sections_joined = {lab: " ".join(sent_list).strip() for lab, sent_list in sections.items()}

    return sections_joined

# =========================
# Main function: transforms PubMed .txt file to .jsonl
# =========================
def preprocess_file_to_jsonl(input_path, output_path, replace_numbers=False, show_progress=True):
    """
    Processes articles file and writes each article as JSONL.
    Each JSON line:
    {"pmid": "...", "sections": {"OBJECTIVE": "...", "METHODS": "..."}}

    Parameters:
    - input_path: path to .txt file
    - output_path: output .jsonl path
    - replace_numbers: anonymize numbers if True
    - show_progress: show progress bar if True
    """
    # Create output directory 
    ensure_dir(output_path.rsplit("/", 1)[0] if "/" in output_path else ".")

    # Open output file for writing
    with open(output_path, "w", encoding="utf-8") as fout:
        # Stream articles from file
        for pmid, article in tqdm(iter_pubmed(input_path), desc=f"Preprocessing {input_path}", disable=not show_progress):
            # Group sentences by sections and clean
            sections = group_sections(article, replace_numbers=replace_numbers)
            # Build JSON object
            record = {"pmid": pmid, "sections": sections}
            # Write JSON line to file
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

# =========================
# Function to get label distribution
# =========================
from collections import Counter
def compute_label_distribution_from_txt(input_path, max_articles=None):
    """
    Counts sentences per label in a .txt file
    - useful for analyzing label distribution
    
    Parameters:
    - max_articles: optional, stop after this number of articles for quick test
    
    Returns:
    - counter: dictionary {label: sentence_count}
    """
    counter = Counter()
    count_articles = 0

    for pmid, article in iter_pubmed(input_path):
        for s in article:
            counter[s.get("label", "UNKNOWN")] += 1
        count_articles += 1
        if max_articles and count_articles >= max_articles:
            break

    return counter

# =========================
# Executable part
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess PubMed RCT txt -> JSONL")
    parser.add_argument("--input", required=True, help="Path to train/dev/test .txt")
    parser.add_argument("--output", required=True, help="Output .jsonl path (ex: data/processed/train_clean.jsonl)")
    parser.add_argument("--replace-numbers", action="store_true", help="Replace numbers with '@'")
    args = parser.parse_args()

    print("Start preprocessing...")
    preprocess_file_to_jsonl(args.input, args.output, replace_numbers=args.replace_numbers)
    print("Done.")