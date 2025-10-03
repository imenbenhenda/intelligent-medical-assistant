# src/preprocessing.py

import re       # Pour expressions régulières (nettoyage du texte)
import json     # Pour écrire les fichiers JSONL
from tqdm import tqdm  # Pour afficher une barre de progression lors du traitement
from src.utils import iter_pubmed, ensure_dir 

# =========================
# Fonction de nettoyage du texte
# =========================
def clean_text(text, lower=True, remove_control_chars=True, keep_unicode_letters=True):
    """
    Nettoie le texte d'une phrase.
    
    Paramètres :
    - text : phrase à nettoyer
    - lower : convertir en minuscules si True
    - remove_control_chars : remplacer retours chariot/tabulations par espaces
    - keep_unicode_letters : conserver lettres accentuées (utile pour vocabulaire médical)
    
    Retour :
    - texte nettoyé, prêt pour le modèle
    """
    if text is None:
        return ""  # Retourne chaîne vide si texte None

    # 1) Conversion en minuscules
    if lower:
        text = text.lower()

    # 2) Supprimer retours à la ligne et tabulations
    if remove_control_chars:
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # 3) Supprimer caractères non désirés
    if keep_unicode_letters:
        # Conserve lettres accentuées, chiffres, ponctuation de base
        text = re.sub(r"[^0-9A-Za-z\u00C0-\u024F\.\,\:\;\(\)\[\]\%\+\-\/\?\!\s]", " ", text)
    else:
        # Sans accent
        text = re.sub(r"[^0-9A-Za-z\.\,\:\;\(\)\[\]\%\+\-\/\?\!\s]", " ", text)

    # 4) Normaliser les espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================
#  anonymisation des nombres
# =========================
def replace_numbers_with_at(text):
    """
    Remplace tous les nombres par '@'
    - utile pour anonymiser chiffres et valeurs numériques
    """
    return re.sub(r"\d+([.,]\d+)?", "@", text)

# =========================
# Fonction pour regrouper les phrases par sections
# =========================
def group_sections(article_sentences, clean_fn=clean_text, replace_numbers=False):
    """
    Transforme un article en dictionnaire de sections.
    
    Paramètres :
    - article_sentences : liste de dicts {'label','text'} pour un article
    - clean_fn : fonction pour nettoyer chaque phrase
    - replace_numbers : si True, remplace tous les nombres par '@'

    Retour :
    - sections_joined : dict {label: "texte complet joint"}
    """
    sections = {}  # dictionnaire temporaire pour stocker les phrases par label

    for s in article_sentences:
        label = s.get("label", "UNKNOWN")  # récupérer label, sinon UNKNOWN
        text = s.get("text", "")
        
        # Nettoyage du texte
        cleaned = clean_fn(text)
        
        # Optionnel : remplacer nombres par '@'
        if replace_numbers:
            cleaned = replace_numbers_with_at(cleaned)
        
        # Ajouter la phrase nettoyée à la section correspondante
        sections.setdefault(label, []).append(cleaned)

    # Joindre toutes les phrases d'une même section en un paragraphe
    sections_joined = {lab: " ".join(sent_list).strip() for lab, sent_list in sections.items()}

    return sections_joined

# =========================
# Fonction principale : transforme un fichier PubMed .txt en .jsonl
# =========================
def preprocess_file_to_jsonl(input_path, output_path, replace_numbers=False, show_progress=True):
    """
    Parcourt le fichier d'articles et écrit chaque article en JSONL.
    Chaque ligne JSON :
    {"pmid": "...", "sections": {"OBJECTIVE": "...", "METHODS": "..."}}

    Paramètres :
    - input_path : chemin du fichier .txt
    - output_path : chemin de sortie .jsonl
    - replace_numbers : anonymiser chiffres si True
    - show_progress : afficher barre de progression si True
    """
    # Créer le dossier de sortie si nécessaire
    ensure_dir(output_path.rsplit("/", 1)[0] if "/" in output_path else ".")

    # Ouvrir fichier de sortie en écriture
    with open(output_path, "w", encoding="utf-8") as fout:
        # Parcourir les articles du fichier en streaming
        for pmid, article in tqdm(iter_pubmed(input_path), desc=f"Preprocessing {input_path}", disable=not show_progress):
            # Regrouper les phrases par sections et nettoyer
            sections = group_sections(article, replace_numbers=replace_numbers)
            # Construire l'objet JSON
            record = {"pmid": pmid, "sections": sections}
            # Écrire ligne JSON dans le fichier
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

# =========================
# Fonction pour obtenir distribution des labels
# =========================
from collections import Counter
def compute_label_distribution_from_txt(input_path, max_articles=None):
    """
    Compte combien de phrases par label dans un fichier .txt
    - utile pour analyser la répartition des labels
    
    Paramètres :
    - max_articles : optionnel, s'arrêter après ce nombre d'articles pour test rapide
    
    Retour :
    - counter : dictionnaire {label: nombre_phrases}
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
# Partie exécutable
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess PubMed RCT txt -> JSONL")
    parser.add_argument("--input", required=True, help="Chemin vers train/dev/test .txt")
    parser.add_argument("--output", required=True, help="Chemin de sortie .jsonl (ex: data/processed/train_clean.jsonl)")
    parser.add_argument("--replace-numbers", action="store_true", help="Remplacer les nombres par '@'")
    args = parser.parse_args()

    print("Start preprocessing...")
    preprocess_file_to_jsonl(args.input, args.output, replace_numbers=args.replace_numbers)
    print("Done.")
