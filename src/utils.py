# src/utils.py

import os  # Module pour gérer les fichiers et dossiers

# =========================
# Fonction pour créer un dossier 
# =========================
def ensure_dir(path):
    """
    Crée le dossier spécifié par `path` s'il n'existe pas déjà.
    - path : chemin du dossier à créer
    """
    os.makedirs(path, exist_ok=True)  # exist_ok=True évite une erreur si le dossier existe déjà

# =========================
# Itérateur pour lire les fichiers PubMed RCT (train/dev/test)
# =========================
def iter_pubmed(file_path, encoding='utf-8'):
    """
    Lit un fichier PubMed RCT ligne par ligne et renvoie chaque article sous forme structurée.
    
    Paramètres :
    - file_path : chemin vers le fichier train/dev/test
    - encoding : encodage du fichier (par défaut UTF-8)

    Renvoie :
    - tuple (pmid, article)
        - pmid : ID PubMed de l'article
        - article : liste de dictionnaires {"label": ..., "text": ...} pour chaque phrase

    Avantages :
    - Lecture en streaming 
    - Gestion des lignes mal formatées ou vides
    """
    
    # Vérification que le fichier existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable: {file_path}")

    pmid = None       # Variable pour stocker l'ID de l'article actuel
    article = []      # Liste pour stocker toutes les phrases de l'article en cours

    # Ouvrir le fichier en lecture
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        for raw_line in f:                # Parcours ligne par ligne
            line = raw_line.strip()       # Supprimer les espaces au début/fin

            # =========================
            # Ligne vide -> fin d'un article
            # =========================
            if not line:
                if pmid is not None and article:  # Si on a un article en cours
                    yield pmid, article          # Retourne l'article
                    article = []                 # Réinitialiser la liste pour le prochain article
                    pmid = None                  # Réinitialiser l'ID
                continue

            # =========================
            # Début d'un nouvel article (ligne commençant par "###")
            # =========================
            if line.startswith("###"):
                # Si on avait un article en cours (rare, mais possible sans ligne vide), on le retourne
                if pmid is not None and article:
                    yield pmid, article
                    article = []

                pmid = line[3:].strip()  # Extraire l'ID après les trois ### et supprimer les espaces
                continue

            # =========================
            # Ligne contenant "LABEL \t TEXTE"
            # =========================
            if "\t" in line:
                label, text = line.split("\t", 1)  # Séparer le label et le texte
                article.append({
                    "label": label.strip(),  # Nettoyer le label
                    "text": text.strip()     # Nettoyer le texte
                })
            else:
                # Gestion robuste : si pas de tabulation, assigner label "UNKNOWN"
                article.append({
                    "label": "UNKNOWN",
                    "text": line
                })

    # =========================
    # En fin de fichier, retourner le dernier article s'il existe
    # =========================
    if pmid is not None and article:
        yield pmid, article
