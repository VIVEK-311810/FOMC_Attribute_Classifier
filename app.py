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
    /* Main app background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 3rem 0 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 3rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 1rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        color: white !important;
    }
    
    .subtitle {
        font-size: 1.4rem !important;
        font-weight: 400 !important;
        opacity: 0.95;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
        color: white !important;
    }
    
    /* Content sections */
    .content-section {
        background: #ffffff;
        padding: 2.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border: 1px solid #e8e8e8;
    }
    
    .intro-text {
        font-size: 1.2rem;
        line-height: 1.8;
        color: #2c3e50 !important;
        text-align: center;
        margin: 2rem auto;
        max-width: 900px;
    }
    
    .intro-text p {
        color: #2c3e50 !important;
    }
    
    .intro-text strong {
        color: #1a1a1a !important;
    }
    
    /* About section styling */
    .about-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 3rem;
        border-radius: 20px;
        margin: 3rem 0;
    }
    
    .about-title {
        color: #2c3e50 !important;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .about-text {
        font-size: 1.1rem;
        line-height: 1.7;
        color: #34495e !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Attributes grid */
    .attributes-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .attribute-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid #e8e8e8;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .attribute-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    
    .attribute-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .attribute-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50 !important;
        margin: 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        margin: 2rem auto;
        display: block;
        width: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Classification results styling */
    .classification-result {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .confidence-high {
        color: #28a745 !important;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffc107 !important;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc3545 !important;
        font-weight: bold;
    }
    
    /* Historical section styling */
    .historical-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 3rem 0;
        border-radius: 2px;
    }
    
    /* Footer styling */
    .footer-info {
        text-align: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 15px;
        margin-top: 3rem;
        color: #6c757d !important;
        font-size: 1rem;
    }
    
    /* Fix text colors for all elements */
    .stMarkdown {
        color: #2c3e50 !important;
    }
    
    .stMarkdown p {
        color: #2c3e50 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #2c3e50 !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem !important;
        }
        
        .subtitle {
            font-size: 1.2rem !important;
        }
        
        .content-section {
            padding: 1.5rem;
        }
        
        .about-section {
            padding: 2rem;
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
    
    st.subheader("🎯 Classification Results")
    
    for key, display_name in display_mapping.items():
        if key in results:
            result = results[key]
            prediction = result.get("prediction", "N/A")
            confidence = result.get("confidence", 0.0)
            
            with st.container():
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**{display_name}:** {prediction}")
                with col2:
                    st.markdown(f"Confidence: {format_confidence(confidence)}", unsafe_allow_html=True)

def home_page():
    """Display the enhanced home page"""
    
    # Main Header Section with centered subheading
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🏛️ FOMC Statement Classifier</h1>
        <div style="text-align: center;">
            <p class="subtitle">A financial-domain BERT model for Federal Reserve (FOMC) document analysis.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Introduction Section with more engaging content
    st.markdown("""
    <div class="content-section">
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #667eea; font-size: 2.2rem; margin-bottom: 1rem;">
                🌟 Decoding the Language of Central Banking 🌟
            </h2>
        </div>
        
        <div class="intro-text">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 2rem; flex-wrap: wrap;">
                <span style="font-size: 3rem; margin-right: 1rem;">💼</span>
                <p style="font-size: 1.3rem; margin: 0; max-width: 600px; color: #2c3e50;">
                    <strong>Federal Open Market Committee (FOMC) statements</strong> represent some of the most consequential communications in global finance, 
                    with each phrase carrying the potential to move markets worth trillions of dollars.
                </p>
            </div>
            
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 2rem; flex-wrap: wrap;">
                <span style="font-size: 3rem; margin-right: 1rem;">🧩</span>
                <p style="font-size: 1.3rem; margin: 0; max-width: 600px; color: #2c3e50;">
                    These carefully crafted documents present a <strong>unique challenge</strong> for financial professionals due to their 
                    technical complexity, diplomatic language, and nuanced policy signals that require expert interpretation.
                </p>
            </div>
            
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 2rem; flex-wrap: wrap;">
                <span style="font-size: 3rem; margin-right: 1rem;">🚀</span>
                <p style="font-size: 1.3rem; margin: 0; max-width: 600px; color: #2c3e50;">
                    Our <strong>advanced AI solution</strong> delivers precise, real-time analysis of these critical policy documents, 
                    transforming complex central bank communications into clear, structured economic intelligence.
                </p>
            </div>
            
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 2rem; flex-wrap: wrap;">
                <span style="font-size: 3rem; margin-right: 1rem;">⚡</span>
                <p style="font-size: 1.3rem; margin: 0; max-width: 600px; color: #2c3e50;">
                    The system identifies key policy signals and classifies them according to established financial taxonomies, 
                    enabling <strong>faster and more accurate decision-making</strong> for traders, analysts, and researchers.
                </p>
            </div>
            
            <div style="background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); padding: 2rem; border-radius: 15px; margin-top: 2rem;">
                <div style="text-align: center;">
                    <h3 style="color: #2c3e50; margin-bottom: 1rem;">🎯 Why This Matters</h3>
                    <p style="font-size: 1.2rem; color: #34495e; margin-bottom: 1rem;">
                        <strong>📊 Market Impact:</strong> FOMC statements can trigger billions in trading volume within minutes
                    </p>
                    <p style="font-size: 1.2rem; color: #34495e; margin-bottom: 1rem;">
                        <strong>⏰ Time Sensitivity:</strong> First-mover advantage in interpreting policy changes is crucial
                    </p>
                    <p style="font-size: 1.2rem; color: #34495e; margin-bottom: 0;">
                        <strong>🔍 Precision Required:</strong> Subtle language changes can signal major policy shifts
                    </p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # About the AI Tool Section
    st.markdown("""
    <div class="about-section">
        <h2 class="about-title">🧠 About the AI Tool</h2>
        <div class="about-text">
            This tool is powered by a <strong>fine-tuned FinBERT model</strong>, a variant of Google's BERT 
            specifically pre-trained on a vast corpus of financial text. We further specialized it 
            using annotated FOMC statements to understand the unique nuances of central bank language.
        </div>
        <div class="about-text">
            Our model identifies key economic signals across <strong>six critical dimensions</strong>:
        </div>
        
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

    # Call to Action
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Enter Classification Tool", type="primary", use_container_width=True):
            st.session_state.page = "classification"
            st.rerun()

    # Footer Information
    st.markdown("""
    <div class="footer-info">
        <strong>Project by:</strong> Vivek Maddula<br>
        <strong>Research Guide:</strong> Dr. Brindha
    </div>
    """, unsafe_allow_html=True)

def classification_page():
    """Display the classification page"""
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
    
    st.title("🎯 FOMC Statement Classification")
    
    # Create two main columns
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("📝 Input Statement")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Type/Paste Text", "Select from Historical Data"],
            horizontal=True
        )
        
        if input_method == "Type/Paste Text":
            user_text = st.text_area(
                "Enter FOMC statement text:",
                height=300,
                placeholder="Paste your FOMC statement text here..."
            )
        else:
            # Historical data selection
            if "df" in st.session_state:
                df = st.session_state.df
                
                # Year filter
                years = sorted(df['year'].unique(), reverse=True)
                selected_year = st.selectbox("Select Year:", years)
                
                # Filter by year
                year_df = df[df['year'] == selected_year]
                
                # Month filter
                months = sorted(year_df['month_year'].unique(), reverse=True)
                selected_month = st.selectbox("Select Month:", months)
                
                # Filter by month
                month_df = year_df[year_df['month_year'] == selected_month]
                
                if not month_df.empty:
                    # Display statement
                    statement_row = month_df.iloc[0]
                    user_text = statement_row['statement_content']
                    
                    st.text_area(
                        f"Statement from {selected_month}:",
                        value=user_text,
                        height=300,
                        disabled=True
                    )
                else:
                    user_text = ""
                    st.warning("No data available for selected period.")
            else:
                user_text = ""
                st.error("Historical data not loaded.")
        
        # Classification button
        if st.button("🔍 Classify Statement", type="primary", use_container_width=True):
            if user_text.strip():
                with st.spinner("Analyzing statement..."):
                    results = classify_statement(user_text)
                    st.session_state.classification_results = results
                    st.session_state.classified_text = user_text
            else:
                st.error("Please enter some text to classify.")
    
    with right_col:
        st.subheader("📊 Classification Results")
        
        if "classification_results" in st.session_state:
            display_classification_results(st.session_state.classification_results)
            
            # Show classified text
            with st.expander("📄 Classified Text", expanded=False):
                st.text_area(
                    "Text that was classified:",
                    value=st.session_state.classified_text,
                    height=200,
                    disabled=True
                )
        else:
            st.info("👈 Enter a statement and click 'Classify Statement' to see results here.")
    
    # Label mappings reference
    st.markdown("---")
    st.subheader("📋 Label Mappings Reference")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Sentiment:**")
        st.markdown("• Positive\n• Neutral\n• Negative")
        
        st.markdown("**Economic Growth:**")
        st.markdown("• UP\n• Down\n• Flat")
    
    with col2:
        st.markdown("**Employment Growth:**")
        st.markdown("• UP\n• Down\n• Flat")
        
        st.markdown("**Inflation:**")
        st.markdown("• UP\n• Down\n• Flat")
    
    with col3:
        st.markdown("**Medium Term Rate:**")
        st.markdown("• Hawk\n• Dove")
        
        st.markdown("**Policy Rate:**")
        st.markdown("• Raise\n• Flat\n• Lower")

def main():
    """Main application function"""
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    # Load models and data with progress indication
    if "models_loaded" not in st.session_state:
        with st.spinner("Loading models and data..."):
            try:
                load_models()
                load_excel_data()
                st.session_state.models_loaded = True
                st.success("✅ Models and data loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")
                return
    
    # Page routing
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "classification":
        classification_page()

if __name__ == "__main__":
    main()

