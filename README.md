# 🏛️ FOMC Statement Classifier

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://fomc-economic-attribute-classifier.streamlit.app/)

An AI-powered tool for analyzing Federal Open Market Committee (FOMC) statements, classifying sentiment and policy signals using fine-tuned financial BERT models.

## 📊 Classification Attributes

Our system analyzes FOMC statements across six key dimensions:

### 1. Sentiment Analysis
**Purpose:** Measures overall tone of the statement  
**Possible Classifications:**
- `Positive` - Optimistic economic outlook
- `Neutral` - Balanced/mixed language
- `Negative` - Pessimistic or cautionary tone

### 2. Economic Growth Outlook
**Purpose:** Assesses projected GDP trajectory  
**Possible Classifications:**
- `Up` - Expecting economic expansion
- `Flat` - Anticipating stable growth
- `Down` - Predicting economic contraction

### 3. Employment Growth Outlook
**Purpose:** Evaluates labor market expectations  
**Possible Classifications:**
- `Up` - Improving employment conditions
- `Flat` - Stable job market outlook
- `Down` - Expected employment slowdown

### 4. Inflation Outlook
**Purpose:** Analyzes price stability projections  
**Possible Classifications:**
- `Up` - Rising inflation expectations
- `Flat` - Stable price projections
- `Down` - Disinflationary pressures

### 5. Medium-Term Rate Outlook
**Purpose:** Predicts 2-5 year rate trajectory  
**Possible Classifications:**
- `Hawkish` - Tightening bias expected
- `Dovish` - Accommodative policy likely

### 6. Policy Rate Direction
**Purpose:** Forecasts near-term Fed actions  
**Possible Classifications:**
- `Raise` - Rate hike anticipated
- `Flat` - Rates expected to hold
- `Lower` - Rate cut projected


## ✨ Features

- **Historical Comparison**: Analyze statement evolution over time
- **Confidence Scoring**: Probability estimates for each classification

## 🛠️ Technical Architecture

```mermaid
graph TD
    A[FOMC Statements] --> B(Preprocessing)
    B --> C[Financial BERT Models]
    C --> D[Sentiment Classifier]
    C --> F[Inflation Classifier]
    C --> E[Economic Growth Classifier]
    C --> G[Employment Growth Classifier]
    C --> H[Policy Rate Classifier]
    C --> I[Medium Term Rate Classifier]
    D --> J[FOMC Statement Classifier]
    E --> J[FOMC Statement Classifier]
    F --> J[FOMC Statement Classifier]
    G --> J[FOMC Statement Classifier]
    I --> J[FOMC Statement Classifier]
```
![App](https://github.com/user-attachments/assets/7ca0eed7-fc56-4ae9-b2ac-16f0f98b26c5)
