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

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    .classification-result {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
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
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
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
    """Display the home page"""
    st.markdown("""
    <style>
        .centered-header {
            text-align: center;
            margin-top: 0;
        }
        .main-title {
            font-size: 3em !important;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.5em !important;
            margin-top: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
      
      # Centered main header
    st.markdown("""
    <div class="centered-header">
        <h1 class="main-title">🏛️ FOMC Statement Classifier</h1>
        <h3 class="subtitle">A financial-domain BERT model for Federal Reserve (FOMC) document analysis.</h3>
    </div>
    """, unsafe_allow_html=True)
  
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h3>
            This project fine-tuned ProsusAI/finBERT using FOMC statements to improve financial NLP tasks.
        </h3>
        <h4>
            ProsusAI → (MLM FineTuning) → fomc_mlm_minutes → (MLM FineTuning) → fomc_mlm_statements → (Classification FineTuning) → FOMC_LLM_VK
        </h4>
        <p style='font-size: 1.1em; margin: 2rem 0;'>
            FOMC Minutes are the detailed records of the Federal Reserve's meetings released late.<br>
            FOMC Statements are concise summaries released immediately after the meeting.<br><br>
            These FOMC data is helpful to guide market expectations and signal the Fed's outlook on Inflation, Economic Growth, and more.<br>
            This model is helpful for classifying these FOMC statements according to 6 attributes:<br>
            <strong>Sentiment, Economic Growth, Employment Growth, Inflation, Medium Term Rate, Policy Rate</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🚀 Enter Classification Tool", type="primary", use_container_width=True):
        st.session_state.page = "classification"
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

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
            "Enter FOMC Statement Text",
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
    
    # Initialize models and data only once
    if "initialized" not in st.session_state:
        with st.spinner("🔄 Loading models and data..."):
            try:
                load_models()
                load_excel_data()
                st.session_state.initialized = True
                st.toast("✅ Models and data loaded successfully!", icon="✅")
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
