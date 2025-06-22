import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from typing import Dict, List, Optional, Any
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
EXCEL_PATH = os.getenv("EXCEL_PATH", "./data/Sentiment&Attributes_Classification.xlsx")
LABEL_COLUMNS = ["Sentiment", "Economic Growth", "Employment Growth",
                 "Inflation", "Medium Term Rate", "Policy Rate"]
MAX_LENGTH = 128

# Hugging Face Model IDs
HUGGINGFACE_MODELS = {
    "Sentiment": "Vk311810/fomc_sentiment_classifier",
    "Economic Growth": "Vk311810/fomc-economic_growth-classifier",
    "Employment Growth": "Vk311810/fomc-employment_growth-classifier",
    "Inflation": "Vk311810/fomc-inflation-classifier",
    "Medium Term Rate": "Vk311810/fomc-medium_rate-classifier",
    "Policy Rate": "Vk311810/fomc-policy_rate-classifier"
}

# Label mappings
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

device = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")

def load_models():
    """Load all models and store them in session state"""
    if "models" not in st.session_state or "tokenizers" not in st.session_state:
        st.session_state.models = {}
        st.session_state.tokenizers = {}
        
        for label, hf_model_id in HUGGINGFACE_MODELS.items():
            try:
                normalized_label = normalize_label(label)
                logger.info(f"Loading model for {label} → {normalized_label}")
                
                # Load model and tokenizer
                model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
                tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
                
                # Store in session state
                st.session_state.models[normalized_label] = model.to(device)
                st.session_state.tokenizers[normalized_label] = tokenizer
                st.session_state.models[normalized_label].eval()
                
                logger.info(f"✅ Successfully loaded model for {label}")
            except Exception as e:
                logger.error(f"❌ Failed to load model for {label}: {str(e)}")
                raise

def load_excel_data():
    """Load Excel data into session state"""
    try:
        if os.path.exists(EXCEL_PATH):
            df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
            df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
            df["year"] = df["Date"].dt.year
            df["month"] = df["Date"].dt.month
            df["month_year"] = df["Date"].dt.strftime("%b %Y")
            df["statement_content"] = df["statement_content"].astype(str)
            st.session_state.df = df
            logger.info(f"Loaded Excel data with {len(df)} records")
            return True
        else:
            logger.warning(f"Excel file not found at {EXCEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"Failed to load Excel file: {str(e)}")
        return False

def classify_statement(text: str) -> Dict[str, Any]:
    """Classify text using loaded models"""
    results = {}
    
    if "models" not in st.session_state or "tokenizers" not in st.session_state:
        logger.error("Models not loaded in session state")
        return {normalize_label(label): {
            "prediction": "N/A",
            "confidence": 0.0,
            "error": "Models not loaded"
        } for label in LABEL_COLUMNS}
    
    for label in LABEL_COLUMNS:
        norm_key = normalize_label(label)
        logger.info(f"Processing {label} (key: {norm_key})")
        
        if norm_key in st.session_state.models and norm_key in st.session_state.tokenizers:
            try:
                tokenizer = st.session_state.tokenizers[norm_key]
                model = st.session_state.models[norm_key]
                
                inputs = tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=MAX_LENGTH
                ).to(device)
                
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
                
                logger.info(f"✅ Classified {label} as {predicted_label} (confidence: {confidence:.2f})")
                
            except Exception as e:
                logger.error(f"Error classifying {label}: {str(e)}")
                results[norm_key] = {
                    "prediction": "Error",
                    "confidence": 0.0,
                    "error": str(e)
                }
        else:
            logger.warning(f"Model not found for {label} (key: {norm_key})")
            results[norm_key] = {
                "prediction": "N/A",
                "confidence": 0.0,
                "error": f"Model for {label} not loaded"
            }
    
    return results

# Page configuration
st.set_page_config(
    page_title="FOMC Statement Classifier",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS for styling with white background and improved design
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #2c3e50; /* Set a default dark text color for the entire app */
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    .main-header {
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 3rem;
        color: white;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
    }

    .main-title {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 1rem !important;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }

    .subtitle {
        font-size: 3rem !important;
        font-weight: 500 !important;
        margin-bottom: 1rem !important;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }

    .content-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 6px 25px rgba(0,0,0,0.08);
        border: 1px solid #e8e8e8;
    }

    .intro-text {
        font-size: 1.2rem;
        line-height: 1.8;
        color: #2c3e50;
        text-align: center;
        margin: 2rem auto;
        max-width: 900px;
    }

    .intro-text p {
        text-align: left;
        margin-bottom: 1.5rem;
    }

    .about-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 3rem;
        border-radius: 20px;
        margin: 3rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    }

    .about-title {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }

    .about-text {
        font-size: 1.1rem;
        line-height: 1.7;
        color: #34495e;
        margin-bottom: 1.5rem;
    }

    .attributes-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .attribute-card {
        background: #fff;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        border: 2px solid #e0e0e0;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }

    .attribute-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }

    .attribute-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }

    .attribute-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 2.5rem;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.25);
        transition: all 0.3s ease;
        margin: 2rem auto 1rem auto;
        display: block;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.35);
    }

    /* Enhanced Classification Page Styling */
    .classification-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }

    .classification-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .classification-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    .stTabs [data-baseweb="tab-list"] button {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .stTabs [data-baseweb="tab-list"] button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        margin: 0;
    }

    /* Tab Content Styling */
    .stTabs [data-baseweb="tab-panel"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border: 1px solid #e8e8e8;
    }

    /* Text Area Styling */
    .stTextArea > div > div > textarea {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 1rem;
        font-size: 1rem;
        line-height: 1.6;
        color: #2c3e50;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
        background: #ffffff;
    }

    /* Classification Results Styling */
    .classification-result {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border: 1px solid #e8e8e8;
    }

    .classification-result h3 {
        color: #667eea;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 700;
    }

    .result-item {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        transition: all 0.3s ease;
    }

    .result-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-color: #667eea;
    }

    .result-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }

    .result-prediction {
        font-size: 1.1rem;
        font-weight: 500;
        color: #495057;
    }

    .confidence-high {
        color: #28a745;
        font-weight: bold;
        font-size: 1.1rem;
    }

    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.1rem;
    }

    .confidence-low {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.1rem;
    }

    /* Select Box Styling */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        color: #2c3e50;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .stSelectbox > div > div:hover {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15);
    }

    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }

    /* Historical Data Section */
    .historical-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 1px solid #e8e8e8;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }

    .historical-section h3 {
        color: #667eea;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 700;
    }

    /* Expander Styling */
    div[data-testid="stExpander"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border: 2px solid #e8e8e8;
        transition: all 0.3s ease;
    }

    div[data-testid="stExpander"]:hover {
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }

    div[data-testid="stExpanderDetails"] {
        background-color: transparent !important;
        padding-top: 1rem;
    }

    /* Table Styling */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border: 1px solid #e8e8e8;
    }

    .stDataFrame table {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-collapse: collapse;
        width: 100%;
    }

    .stDataFrame th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        padding: 1rem;
        text-align: left;
        font-size: 1.1rem;
    }

    .stDataFrame td {
        padding: 1rem;
        border-bottom: 1px solid #e8e8e8;
        color: #2c3e50;
        font-size: 1rem;
    }

    .stDataFrame tr:last-child td {
        border-bottom: none;
    }

    .stDataFrame tbody tr:hover {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f2ff 100%);
    }

    /* Back Button Styling */
    .back-button {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(108, 117, 125, 0.3);
    }

    .back-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(108, 117, 125, 0.4);
    }

    hr {
        border: none;
        height: 2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 3rem 0;
        border-radius: 2px;
    }

    .footer-info {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        margin-top: 3rem;
        color: #6c757d;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    .stMarkdown {
        margin-bottom: 0;
    }

    @media (max-width: 768px) {
        .main-title {
            font-size: 2.4rem !important;
        }

        .subtitle {
            font-size: 1.1rem !important;
            padding: 0 1rem;
        }

        .content-section {
            padding: 1.5rem;
        }

        .about-section {
            padding: 2rem;
        }

        .attribute-card {
            padding: 1rem;
        }

        .stButton > button {
            width: 100% !important;
        }

        .classification-header h1 {
            font-size: 2rem !important;
        }

        .classification-header p {
            font-size: 1rem !important;
        }

        .stTabs [data-baseweb="tab-panel"] {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)


def format_confidence(confidence: float) -> str:
    """Format confidence score with color coding"""
    percentage = confidence * 100
    if percentage >= 80:
        return f'<span class="confidence-high">{percentage:.1f}%</span>'
    elif percentage >= 60:
        return f'<span class="confidence-medium">{percentage:.1f}%</span>'
    else:
        return f'<span class="confidence-low">{percentage:.1f}%</span>'

def display_classification_results(results: Dict):
    """Display classification results in a formatted way"""
    if not results:
        st.warning("No classification results available.")
        return
    
    # Map API response keys to display names
    display_mapping = {
        "sentiment": "Sentiment",
        "economic_growth": "Economic Growth", 
        "employment_growth": "Employment Growth",
        "inflation": "Inflation",
        "medium_term_rate": "Medium Term Rate",
        "policy_rate": "Policy Rate"
    }
    
    # Create a styled results container
    st.markdown("""
    <div class="classification-result">
        <h3>🎯 Classification Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    for key, display_name in display_mapping.items():
        if key in results:
            result = results[key]
            prediction = result.get("prediction", "N/A")
            confidence = result.get("confidence", 0.0)
            
            # Create styled result item
            st.markdown(f"""
            <div class="result-item">
                <div class="result-label">{display_name}</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="result-prediction">Prediction: <strong>{prediction}</strong></div>
                    <div>Confidence: {format_confidence(confidence)}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def home_page():
    """Display the enhanced home page"""

    # Main Header Section
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🏛️ FOMC Statement Classifier</h1>
        <h3 class="subtitle">A financial BERT model for Federal Reserve (FOMC) document analysis.</h3>
    </div>
    """, unsafe_allow_html=True)


    # Introduction Section
    st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
        }
    
        .content-section {
            padding: 2rem 4rem;
        }
    
        .section-block {
            display: flex;
            align-items: flex-start;
            gap: 1.5rem;
            margin-bottom: 2rem;
            font-size: 1.25rem;
            line-height: 1.6;
        }
    
        .section-block span {
            font-size: 2.5rem;
            flex-shrink: 0;
        }
    
        .section-text {
            flex-grow: 1;
            max-width: 100%;
        }
    
        .highlight-box {
            background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-top: 2rem;
        }
    
        .highlight-box h3 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 1rem;
        }
    
        .highlight-box p {
            font-size: 1.15rem;
            color: #34495e;
            text-align: center;
            margin: 0.5rem 0;
        }
    
        h2.title {
            color: #667eea;
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 2.5rem;
        }
    </style>
    
    <div class="content-section">
        <h2 class="title">🌟 Decoding the Language of Central Banking 🌟</h2>    
        <div class="section-block">
            <span>💼</span>
            <div class="section-text">
                <strong>Federal Open Market Committee (FOMC) statements</strong> represent some of the most consequential communications in global finance, 
                with each phrase carrying the potential to move markets worth trillions of dollars.
            </div>
        </div>    
        <div class="section-block">
            <span>🧩</span>
            <div class="section-text">
                These carefully crafted documents present a <strong>unique challenge</strong> for financial professionals due to their 
                technical complexity, diplomatic language, and nuanced policy signals that require expert interpretation.
            </div>
        </div>
        <div class="section-block">
            <span>🚀</span>
            <div class="section-text">
                Our <strong>advanced AI solution</strong> delivers precise, real-time analysis of these critical policy documents, 
                transforming complex central bank communications into clear, structured economic intelligence.
            </div>
        </div>
        <div class="section-block">
            <span>⚡</span>
            <div class="section-text">
                The system identifies key policy signals and classifies them according to established financial taxonomies, 
                enabling <strong>faster and more accurate decision-making</strong> for traders, analysts, and researchers.
            </div>
        </div>    
        <div class="highlight-box">
            <h3>🎯 Why This Matters</h3>
            <p><strong>📊 Market Impact:</strong> FOMC statements can trigger billions in trading volume within minutes</p>
            <p><strong>⏰ Time Sensitivity:</strong> First-mover advantage in interpreting policy changes is crucial</p>
            <p><strong>🔍 Precision Required:</strong> Subtle language changes can signal major policy shifts</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # About the AI Tool Section
    st.markdown("""
    <div class="about-section">
        <h2 class="about-title">🧠 About the AI Tool</h2>
        <p class="about-text">
            This tool is powered by a <strong>fine-tuned FinBERT model</strong>, a variant of Google's BERT specifically pre-trained on a vast corpus of financial text. 
            We further specialized it using annotated FOMC statements to understand the unique nuances of central bank language.
        </p>
        <p class="about-text">
            Our model identifies key economic signals across <strong>six critical dimensions</strong>:
        </p>
        <div class="attributes-grid">
            <div class="attribute-card">
                <div class="attribute-icon">📊</div>
                <div class="attribute-title">Sentiment</div>
            </div>
            <div class="attribute-card">
                <div class="attribute-icon">📈</div>
                <div class="attribute-title">Economic Growth</div>
            </div>
            <div class="attribute-card">
                <div class="attribute-icon">👷</div>
                <div class="attribute-title">Employment Growth</div>
            </div>
            <div class="attribute-card">
                <div class="attribute-icon">💹</div>
                <div class="attribute-title">Inflation</div>
            </div>
            <div class="attribute-card">
                <div class="attribute-icon">🦅</div>
                <div class="attribute-title">Medium Term Rate</div>
            </div>
            <div class="attribute-card">
                <div class="attribute-icon">⚖️</div>
                <div class="attribute-title">Policy Rate</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Button to enter classification tool
    if st.button("➡️ Enter Classification Tool"): 
        st.session_state.page = "classification"
        st.experimental_rerun()

    # Footer
    st.markdown("""
    <div class="footer-info">
        <p>&copy; 2025 FOMC Statement Classifier. All rights reserved.</p>
        <p>Built with ❤️ using Streamlit and Hugging Face Transformers.</p>
    </div>
    """, unsafe_allow_html=True)

def classification_page():
    """Display the enhanced classification page"""
    
    # Enhanced Header for Classification Page
    st.markdown("""
    <div class="classification-header">
        <h1>🎯 FOMC Statement Classification</h1>
        <p>Analyze the sentiment and attributes of Federal Reserve communications with advanced AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Back button with custom styling
    if st.button("⬅️ Back to Home", key="back_home"): 
        st.session_state.page = "home"
        st.experimental_rerun()

    # Enhanced tabs with better styling
    tab1, tab2, tab3 = st.tabs(["🔍 Classify New Statement", "📚 Browse Historical Data", "📋 Label Mappings"])

    with tab1:
        st.markdown("""
        <div class="historical-section">
            <h3>🔍 Classify a New FOMC Statement</h3>
        </div>
        """, unsafe_allow_html=True)
        
        user_input = st.text_area(
            "Enter FOMC Statement Text Here:", 
            height=200,
            placeholder="Paste your FOMC statement text here for AI-powered analysis..."
        )
        
        if st.button("🚀 Classify Statement", key="classify_new"): 
            if user_input:
                with st.spinner("🤖 Analyzing statement with AI models..."):
                    results = classify_statement(user_input)
                    display_classification_results(results)
            else:
                st.warning("⚠️ Please enter some text to classify.")

    with tab2:
        st.markdown("""
        <div class="historical-section">
            <h3>📚 Browse Historical FOMC Statements</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if "df" in st.session_state:
            df = st.session_state.df
            
            # Enhanced filters with better styling
            col_year, col_month = st.columns(2)
            with col_year:
                selected_year = st.selectbox(
                    "📅 Select Year", 
                    options=["All"] + sorted(df["year"].unique().tolist(), reverse=True)
                )
            with col_month:
                # Filter months based on selected year
                if selected_year != "All":
                    months_in_year = sorted(df[df["year"] == selected_year]["month_year"].unique().tolist(), 
                                            key=lambda x: pd.to_datetime(x, format="%b %Y"))
                else:
                    months_in_year = sorted(df["month_year"].unique().tolist(), 
                                            key=lambda x: pd.to_datetime(x, format="%b %Y"))
                selected_month_year = st.selectbox("📆 Select Month", options=["All"] + months_in_year)

            filtered_df = df.copy()
            if selected_year != "All":
                filtered_df = filtered_df[filtered_df["year"] == selected_year]
            if selected_month_year != "All":
                filtered_df = filtered_df[filtered_df["month_year"] == selected_month_year]
            
            if not filtered_df.empty:
                st.success(f"📊 Found {len(filtered_df)} statements.")
                selected_statement = st.selectbox(
                    "📄 Select a statement to view and classify:",
                    options=filtered_df["month_year"].tolist(),
                    format_func=lambda x: f"{x} - {filtered_df[filtered_df['month_year'] == x]['Date'].dt.strftime('%Y-%m-%d').iloc[0]}"
                )
                
                if selected_statement:
                    statement_content = filtered_df[filtered_df["month_year"] == selected_statement]["statement_content"].iloc[0]
                    
                    # Enhanced expander for statement content
                    with st.expander("📖 View Statement Content", expanded=True):
                        st.text_area("Statement Content:", statement_content, height=300, disabled=True)
                    
                    if st.button("🚀 Classify Selected Statement", key="classify_historical"): 
                        with st.spinner("🤖 Analyzing historical statement..."):
                            results = classify_statement(statement_content)
                            display_classification_results(results)
            else:
                st.info("ℹ️ No statements found for the selected filters.")
        else:
            st.warning("⚠️ Historical data not loaded. Please ensure the Excel file is in the correct path.")

    with tab3:
        st.markdown("""
        <div class="historical-section">
            <h3>📋 Label Mappings Reference</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("This section explains the mapping of model outputs to human-readable labels.")
        
        # Create a more visually appealing table
        for label_type, mapping in label_maps.items():
            st.markdown(f"### 🏷️ {label_type} Mappings")
            
            # Convert to DataFrame for better display
            mapping_df = pd.DataFrame([
                {"Label": key, "Internal ID": value} 
                for key, value in mapping.items()
            ])
            st.dataframe(mapping_df, use_container_width=True)
            st.markdown("---")

# Main app logic
if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "home"

    # Display loading message only once at the beginning
    if "models_loaded" not in st.session_state:
        with st.spinner("🚀 Loading AI models and historical data..."):
            load_models()
            if load_excel_data():
                st.success("✅ Models and data loaded successfully!")
            else:
                st.error("❌ Failed to load historical data. Please check the Excel file path.")
            st.session_state.models_loaded = True

    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "classification":
        classification_page()

