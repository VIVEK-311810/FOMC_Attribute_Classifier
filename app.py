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
    
    .attributes-grid a:link{
      text-decoration:none;
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
        cursor:pointer;
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
        text-decoration: none;
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

    .classification-result {
        background: #ffffff;
        padding: 1.5rem;
        color: #2c3e50;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }

    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }

    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }

    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }

    .historical-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }

    hr {
        border: none;
        height: 2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 3rem 0;
        border-radius: 2px;
    }

    .stMarkdown {
        margin-bottom: 0;
    }

    .stDataFrame {
        background-color: white !important;
        color: black !important;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    .stDataFrame table {
        background-color: white !important;
        color: black !important;
        border-collapse: collapse;
        width: 100%;
    }
    
    .stDataFrame th {
        background-color: white !important;
        color: black !important;
        font-weight: bold;
        padding: 12px 15px;
        text-align: left;
        border-bottom: 2px solid #a0d0d0;
    }
    
    .stDataFrame td {
        background-color: white !important;
        color: black !important;
        padding: 10px 15px;
        border-bottom: 1px solid #c1e9e9;
    }
    
    .stDataFrame tr:last-child td {
        border-bottom: none;
    }
    
    .stDataFrame tbody tr:hover {
        background-color: #f0f8ff !important;  /* Light blue hover color */
    }

    div[data-testid="stExpander"] {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);      
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #d1e0f0;
    }

    div[data-testid="stExpanderDetails"] {
        background-color: transparent !important;
    }

    div[data-testid="stMarkdownContainer"] {
        background-color: transparent !important;
        color: black;
    }

    div[data-testid="stTextAreaRootElement"] div {
        background-color: white !important;
        color: black;
    }

    div[data-testid="stTextAreaRootElement"] div textarea {
        background-color: white !important;
        color: black;
    }
    div[data-testid="stTextAreaRootElement"] div textarea::placeholder{
        background-color: white !important;
        color: black;
    }

    .stSelectbox div div{
    background-color: white;
    color: black;
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

  
    # Call-to-Action Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Enter Classification Tool", type="primary", use_container_width=True):
            st.session_state.page = "classification"
            st.rerun()


    # About Section
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
        <div class="about-text" style="font-size: 1rem; margin-top: -1rem; margin-bottom: 1rem;">
            <i class="fas fa-mouse-pointer"></i> Click any attribute below to view the specific model on Hugging Face
        </div>
        <div class="attributes-grid" >
            <a href="https://huggingface.co/Vk311810/fomc_sentiment_classifier">
            <div class="attribute-card" >
                <div class="attribute-icon">📊</div>
                <div class="attribute-title">Sentiment</div>
            </div>
            </a>
            <a href="https://huggingface.co/Vk311810/fomc-economic_growth-classifier">
            <div class="attribute-card">
                <div class="attribute-icon">📈</div>
                <div class="attribute-title">Economic Growth</div>
            </div>
            </a>
            <a href="https://huggingface.co/Vk311810/fomc-employment_growth-classifier">
            <div class="attribute-card">
                <div class="attribute-icon">👷</div>
                <div class="attribute-title">Employment Growth</div>
            </div>
            </a>
            <a href="https://huggingface.co/Vk311810/fomc-inflation-classifier">
            <div class="attribute-card">
                <div class="attribute-icon">💹</div>
                <div class="attribute-title">Inflation</div>
            </div>
            </a>
            <a href="https://huggingface.co/Vk311810/fomc-medium_rate-classifier">
            <div class="attribute-card">
                <div class="attribute-icon">🦅</div>
                <div class="attribute-title">Medium Term Rate</div>
            </div>
            </a>
            <a href="https://huggingface.co/Vk311810/fomc-policy_rate-classifier">
            <div class="attribute-card">
                <div class="attribute-icon">⚖️</div>
                <div class="attribute-title">Policy Rate</div>
            </div>
            </a>
        </div>
        
        
    </div>
    """, unsafe_allow_html=True)




def classification_page():
    """Display the classification page"""
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
    
    st.title("🎯 FOMC Statement Classification")
    
    # Create two main columns
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("📝 Input")
        
        # Historical data section
        with st.expander("📊 Select from Historical Data", expanded=False):
            
            # Get available years
            if st.session_state.df is not None:
                years = sorted(st.session_state.df["year"].unique())
                
                selected_year = st.selectbox("Select Year", years, key="year_select")
                
                if selected_year:
                    # Get months for selected year
                    months_df = st.session_state.df[st.session_state.df["year"] == selected_year]
                    months_data = months_df[["month", "month_year"]].drop_duplicates().sort_values(by="month").to_dict(orient="records")
                    
                    if months_data:
                        month_options = [(m["month_year"], m["month"]) for m in months_data]
                        
                        selected_month_name = st.selectbox(
                            "Select Month", 
                            [name for name, _ in month_options],
                            key="month_select"
                        )
                        
                        if selected_month_name:
                            # Get month number
                            selected_month_num = next(num for name, num in month_options if name == selected_month_name)
                            
                            # Get statements for selected year/month
                            statements = st.session_state.df[
                                (st.session_state.df["year"] == selected_year) &
                                (st.session_state.df["month"] == selected_month_num)
                            ].to_dict(orient="records")
                            
                            if statements:
                                statement_options = [
                                    f"{stmt['month_year']} - {stmt['statement_content'][:50]}..."
                                    for stmt in statements
                                ]
                                
                                selected_statement_idx = st.selectbox(
                                    "Select Statement",
                                    range(len(statement_options)),
                                    format_func=lambda x: statement_options[x],
                                    key="statement_select"
                                )
                                
                                if st.button("📥 Load Selected Statement"):
                                    selected_statement = statements[selected_statement_idx]
                                    st.session_state.loaded_text = selected_statement["statement_content"]
                                    
                                    # Prepare actual values for display
                                    actual_vals = {}
                                    for col in LABEL_COLUMNS:
                                        if col in selected_statement and pd.notna(selected_statement[col]):
                                            actual_vals[col] = selected_statement[col]
                                    st.session_state.actual_values = actual_vals
                                    
                                    st.success("Statement loaded!")
                                    st.rerun()
                            else:
                                st.info("No statements available for this period.")
                    else:
                        st.warning("No months available for this year.")
            else:
                st.warning("Historical data not available. Please ensure the Excel file is correctly loaded.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Text input
        default_text = st.session_state.get("loaded_text", "")
        text_input = st.text_area(
            "Enter Text Here",
            value=default_text,
            height=200,
            placeholder="Paste FOMC statement text here or select from historical data above...",
            key="text_input"
        )
        
        # Classification button
        if st.button("🎯 Classify Statement", type="primary", use_container_width=True):
            if text_input.strip():
                with st.spinner("Classifying statement..."):
                    results = classify_statement(text_input)
                    if results:
                        st.session_state.classification_results = results
                        st.success("Classification completed!")
                        st.rerun()
            else:
                st.warning("Please enter some text to classify.")
    
    with right_col:
        st.subheader("📊 Results")
        
        # Display classification results
        if "classification_results" in st.session_state:
            display_classification_results(st.session_state.classification_results)
            
            # Show actual values if available
            if "actual_values" in st.session_state and st.session_state.actual_values:
                st.subheader("📋 Actual Values (from Historical Data)")
                actual_values = st.session_state.actual_values
                
                for category, actual_value in actual_values.items():
                    st.markdown(f"**{category}:** {actual_value}")
        else:
            st.info("Results will appear here after classification...")
        
        # Label mappings reference
        st.subheader("📚 Label Mappings Reference")
        
        # Get label mappings directly
        mapping_data = []
        for category, labels in label_maps.items():
            labels_str = ", ".join(labels.keys())
            mapping_data.append({"Category": category, "Labels": labels_str})
        
        df_mappings = pd.DataFrame(mapping_data)
        st.dataframe(df_mappings, use_container_width=True, hide_index=True)

def main():
    """Main application logic"""
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "home"

    st.markdown("""
    <style>
        .stSpinner > div > div {
            color: black !important;
        }
    </style>
    """, unsafe_allow_html=True)
    # Initialize models and data only once
    if "initialized" not in st.session_state:
        with st.spinner("🔄 Loading models and data..."):
            try:
                load_models()
                load_excel_data()
                st.session_state.initialized = True
                st.toast("✅ Models and data loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                return

    # Route to appropriate page
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "classification":
        classification_page()

if __name__ == "__main__":
    main()
