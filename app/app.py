# app/app.py - VERSION CORRECTE
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import torch
import os

app = Flask(__name__)

# Chemin ABSOLU vers votre modèle
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "biobert-classifier")

print(f"🔄 Chargement du modèle depuis: {MODEL_PATH}")

# Vérifier que le modèle existe
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERREUR: Modèle introuvable à {MODEL_PATH}")
    print("📁 Contenu du dossier models/:")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            print(f"   - {item}")
    exit(1)

# Charger le modèle avec le chemin ABSOLU
try:
    classifier = pipeline(
        "text-classification",
        model=MODEL_PATH,  # Chemin absolu
        tokenizer=MODEL_PATH,
        device=-1  # Forcer CPU pour éviter les problèmes
    )
    print("✅ Modèle chargé avec succès!")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_article():
    try:
        # Récupérer le texte de l'article
        article_text = request.form.get('text', '')
        
        if not article_text.strip():
            return jsonify({"error": "Veuillez entrer un texte d'article"}), 400
        
        # Découper l'article en phrases (simplifié)
        sentences = [s.strip() for s in article_text.split('.') if s.strip()]
        
        # Analyser chaque phrase
        results = []
        for sentence in sentences[:20]:  # Limiter pour la démo
            if len(sentence) > 10:  # Ignorer les phrases trop courtes
                try:
                    classification = classifier(sentence[:512])[0]  # Limiter la longueur
                    results.append({
                        "sentence": sentence,
                        "label": classification['label'],
                        "confidence": round(classification['score'] * 100, 2)
                    })
                except Exception as e:
                    print(f"⚠️ Erreur sur la phrase: {sentence[:50]}... - {e}")
                    continue
        
        # Générer un résumé structuré
        summary = generate_summary(results)
        
        return jsonify({
            "success": True,
            "analysis": results,
            "summary": summary,
            "total_sentences": len(results)
        })
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        return jsonify({"error": f"Erreur lors de l'analyse: {str(e)}"}), 500

def generate_summary(analysis_results):
    """Génère un résumé structuré à partir des analyses"""
    sections = {
        "OBJECTIVE": [],
        "BACKGROUND": [],
        "METHODS": [],
        "RESULTS": [],
        "CONCLUSIONS": []
    }
    
    for item in analysis_results:
        label = item["label"]
        if label in sections:
            sections[label].append(item["sentence"])
    
    # Créer le résumé
    summary_parts = []
    for section_type, sentences in sections.items():
        if sentences:
            # Prendre les 2 premières phrases de chaque section
            preview = ' '.join(sentences[:2])
            if len(preview) > 150:  # Limiter la longueur
                preview = preview[:147] + "..."
            summary_parts.append(f"**{section_type}**: {preview}")
    
    return "\n\n".join(summary_parts) if summary_parts else "Aucune section identifiée."

if __name__ == '__main__':
    print("🌐 Démarrage du serveur Flask...")
    print("📍 Accédez à: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)