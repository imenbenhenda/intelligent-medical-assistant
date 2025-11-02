# app/app.py
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import torch
import os

app = Flask(__name__)


MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "biobert-classifier")

print(f"Loading model from: {MODEL_PATH}")


if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Content of models/ folder:")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            print(f"   - {item}")
    exit(1)

# Load model 
try:
    classifier = pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=-1  # Use CPU
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_article():
    try:
        # Get article text
        article_text = request.form.get('text', '')
        
        if not article_text.strip():
            return jsonify({"error": "Please enter article text"}), 400
        
        # Split article into sentences
        sentences = [s.strip() for s in article_text.split('.') if s.strip()]
        
        # Analyze each sentence
        results = []
        for sentence in sentences[:20]:  # Limit for demo
            if len(sentence) > 10:  # Ignore short sentences
                try:
                    classification = classifier(sentence[:512])[0]  # Limit length
                    results.append({
                        "sentence": sentence,
                        "label": classification['label'],
                        "confidence": round(classification['score'] * 100, 2)
                    })
                except Exception as e:
                    print(f"Error on sentence: {sentence[:50]}... - {e}")
                    continue
        
        # Generate structured summary
        summary = generate_summary(results)
        
        return jsonify({
            "success": True,
            "analysis": results,
            "summary": summary,
            "total_sentences": len(results)
        })
        
    except Exception as e:
        print(f"General error: {e}")
        return jsonify({"error": f"Analysis error: {str(e)}"}), 500

def generate_summary(analysis_results):
    """Generate structured summary from analysis results"""
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
    
    # Create summary
    summary_parts = []
    for section_type, sentences in sections.items():
        if sentences:
            # Take first 2 sentences from each section
            preview = ' '.join(sentences[:2])
            if len(preview) > 150:
                preview = preview[:147] + "..."
            summary_parts.append(f"**{section_type}**: {preview}")
    
    return "\n\n".join(summary_parts) if summary_parts else "No sections identified."

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)