# src/utils.py

import os

# =========================
# Function to create directory
# =========================
def ensure_dir(path):
    """
    Creates the directory specified by `path` if it doesn't exist.
    - path: path of directory to create
    """
    os.makedirs(path, exist_ok=True)

# =========================
# Iterator for reading PubMed RCT files 
# =========================
def iter_pubmed(file_path, encoding='utf-8'):
    """
    Reads a PubMed RCT file line by line and returns each article in structured format.
    
    Parameters:
    - file_path: path to train/dev/test file
    - encoding: file encoding (default UTF-8)

    Returns:
    - tuple (pmid, article)
        - pmid: PubMed ID of the article
        - article: list of dictionaries {"label": ..., "text": ...} for each sentence

    Advantages:
    - Streaming reading
    - Handles malformed or empty lines
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    pmid = None       # Variable to store current article ID
    article = []      # List to store all sentences of current article

    # Open file for reading
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        for raw_line in f:                # Read line by line
            line = raw_line.strip()       # Remove leading/trailing spaces

            # =========================
            # Empty line -> end of article
            # =========================
            if not line:
                if pmid is not None and article:  # If we have current article
                    yield pmid, article           # Return the article
                    article = []                  # Reset list for next article
                    pmid = None                   # Reset ID
                continue

            # =========================
            # Start of new article (line starting with "###")
            # =========================
            if line.startswith("###"):
                # If we had current article , return it
                if pmid is not None and article:
                    yield pmid, article
                    article = []

                pmid = line[3:].strip()  # Extract ID after three ### and remove spaces
                continue

            # =========================
            # Line containing "LABEL \t TEXT"
            # =========================
            if "\t" in line:
                label, text = line.split("\t", 1)  # Separate label and text
                article.append({
                    "label": label.strip(),  # Clean label
                    "text": text.strip()     # Clean text
                })
            else:
                # Robust handling: if no tabulation, assign "UNKNOWN" label
                article.append({
                    "label": "UNKNOWN",
                    "text": line
                })

    # =========================
    # At end of file, return last article if exists
    # =========================
    if pmid is not None and article:
        yield pmid, article