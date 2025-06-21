import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
EXCEL_PATH = os.getenv("EXCEL_PATH", "./data/Sentiment&Attributes_Classification.xlsx")
LABEL_COLUMNS = ["Sentiment", "Economic Growth", "Employment Growth", "Inflation", "Medium Term Rate", "Policy Rate"]
MAX_LENGTH = 128

HUGGINGFACE_MODELS = {
    "Sentiment": "Vk311810/fomc_sentiment_classifier",
    "Economic Growth": "Vk311810/fomc-economic_growth-classifier",
    "Employment Growth": "Vk311810/fomc-employment_growth-classifier",
    "Inflation": "Vk311810/fomc-inflation-classifier",
    "Medium Term Rate": "Vk311810/fomc-medium_rate-classifier",
    "Policy Rate": "Vk311810/fomc-policy_rate-classifier"
}

label_maps = {
    "Sentiment": {"Positive": 0, "Neutral": 1, "Negative": 2},
    "Economic Growth": {"UP": 0, "Down": 1, "Flat": 2},
    "Employment Growth": {"UP": 0, "Down": 1, "Flat": 2},
    "Inflation": {"UP": 0, "Down": 1, "Flat": 2},
    "Medium Term Rate": {"Hawk": 0, "Dove": 1},
    "Policy Rate": {"Raise": 0, "Flat": 1, "Lower": 2}
}

reverse_label_maps = {
    label: {v: k for k, v in mapping.items()}
    for label, mapping in label_maps.items()
}

models = {}
tokenizers = {}
df = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")

def load_excel_data():
    global df
    try:
        if os.path.exists(EXCEL_PATH):
            df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
            df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
            df["year"] = df["Date"].dt.year
            df["month"] = df["Date"].dt.month
            df["month_year"] = df["Date"].dt.strftime("%b %Y")
            df["statement_content"] = df["statement_content"].astype(str)
            logger.info(f"✅ Loaded Excel data with {len(df)} records")
            return True
        else:
            logger.warning(f"❌ Excel file not found at {EXCEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to load Excel file: {str(e)}")
        return False

def load_models():
    global models, tokenizers
    for label, hf_model_id in HUGGINGFACE_MODELS.items():
        try:
            norm_key = normalize_label(label)
            models[norm_key] = AutoModelForSequenceClassification.from_pretrained(hf_model_id).to(device).eval()
            tokenizers[norm_key] = AutoTokenizer.from_pretrained(hf_model_id)
            logger.info(f"✅ Loaded model for {label} as {norm_key}")
        except Exception as e:
            logger.error(f"❌ Failed to load model for {label}: {str(e)}")

def classify_statement(text: str) -> Dict[str, Any]:
    results = {}
    for label in LABEL_COLUMNS:
        norm_key = normalize_label(label)
        if norm_key in models and norm_key in tokenizers:
            tokenizer = tokenizers[norm_key]
            model = models[norm_key]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]
            predicted_class_id = torch.argmax(probabilities).item()
            predicted_label = reverse_label_maps[label][predicted_class_id]
            confidence = probabilities[predicted_class_id].item()
            results[norm_key] = {
                "prediction": predicted_label,
                "confidence": confidence
            }
        else:
            results[norm_key] = {
                "prediction": "N/A",
                "confidence": 0.0,
                "error": f"Model for {label} not loaded"
            }
    logger.info(f"🔍 Classification results: {results}")
    return results

def startup():
    logger.info("🚀 Starting FOMC Classifier Streamlit App...")
    logger.info(f"Device: {device}")
    logger.info(f"Excel Path: {EXCEL_PATH}")
    load_models()
    load_excel_data()
    loaded_keys = list(models.keys())
    logger.info(f"✅ Loaded models for: {', '.join(loaded_keys)}")

# Set up Streamlit UI (simplified for clarity)
st.set_page_config(page_title="FOMC Classifier", layout="wide")
st.title("🏛️ FOMC Statement Classifier")

if "models_loaded" not in st.session_state:
    with st.spinner("🔄 Loading models and data..."):
        startup()
        st.session_state.models_loaded = True
        st.session_state.df = df
        st.success("✅ All models loaded successfully!")

# Text input box
text = st.text_area("📝 Paste FOMC statement here:", height=200)
if st.button("🎯 Classify"):
    if text.strip():
        with st.spinner("Classifying..."):
            result = classify_statement(text)
            st.subheader("📊 Classification Results")
            for key, res in result.items():
                label = key.replace("_", " ").title()
                prediction = res["prediction"]
                conf = res["confidence"] * 100
                st.write(f"**{label}** → {prediction} ({conf:.1f}%)")
    else:
        st.warning("❗ Please enter some text to classify.")
