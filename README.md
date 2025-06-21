# 🏛️ FOMC Statement Classifier

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://fomc-economic-attribute-classifier.streamlit.app/)

An AI-powered tool for analyzing Federal Open Market Committee (FOMC) statements, classifying sentiment and policy signals using fine-tuned financial BERT models.


## ✨ Features

- **Sentiment Analysis**: Positive/Neutral/Negative classification
- **Policy Signals**: Hawkish/Dovish stance detection
- **Economic Indicators**:
  - Economic growth outlook (Up/Down/Flat)
  - Inflation projections
  - Employment trends
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
