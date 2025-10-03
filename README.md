# 🏥 Intelligent Assistant for Medical Literature

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green)](https://flask.palletsprojects.com/)
[![Transformers](https://img.shields.io/badge/%20Transformers-Latest-orange)](https://huggingface.co/transformers)

## 🚀 Live Demo
**Try it here:** https://github.com/imenbenhenda/intelligent-medical-assistant

## 📋 Project Overview
An intelligent LLM-based system that helps medical researchers quickly analyze scientific articles by automatically extracting key information and generating structured summaries.

**Key Features:**
- 🔬 Automatic extraction of medical paper sections (Objective/Methods/Results/Conclusions)
- 📝 AI-powered summarization of complex medical literature  
- 🎯 92.5% accuracy in section classification
- ⚡ Real-time analysis with Flask web interface

## 🏗️ Project Structure
MedicalLiteratureAssistant/
├── app/ # Flask web interface
├── src/ # Core ML pipeline
├── notebooks/ # Data exploration & analysis
├── data/ # Dataset configuration
└── models/ # Model configuration

## 🛠️ Installation & Usage

### 1. Clone Repository
```bash
git clone https://github.com/your-username/medical-literature-assistant.git
cd medical-literature-assistant
### 2. Install Dependencies
pip install -r requirements.txt
### 3. Download Dataset
# Download PubMed 200k RCT dataset from:
# https://github.com/Franck-Dernoncourt/pubmed-rct
# Place in data/PubMed_200k_RCT/
### 4. Launch Application
python app/app.py
# Visit http://localhost:5000
📊 Model Performance

Accuracy: 92.46%
F1-Score: 92.36%
Precision: 97.02% on Methods, 97.61% on Results
Recall: 92.46%

🎯 Technical Highlights

Fine-tuned BioBERT on PubMed 200k RCT dataset
Custom preprocessing pipeline for medical text
Real-time Flask interface with responsive design
Batch processing for efficient analysis

📁 Dataset
PubMed 200k RCT: 200,000 medical abstracts
Sequential sentence classification for structured extraction
Publicly available on GitHub
👨‍💻 Author
Imen Ben Henda - Computer Engineering Student
