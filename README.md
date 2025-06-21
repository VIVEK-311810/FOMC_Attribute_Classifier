# 🏛️ FOMC Statement Classifier

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://fomc-classifier.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An AI-powered tool for analyzing Federal Open Market Committee (FOMC) statements, classifying sentiment and policy signals using fine-tuned financial BERT models.

![App Screenshot](https://example.com/path/to/screenshot.png) <!-- Replace with actual screenshot -->

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
    C --> E[Economic Growth Classifier]
    C --> F[Inflation Classifier]
    C --> G[Policy Rate Classifier]
    D --> H[Visualization Dashboard]
    E --> H
    F --> H
    G --> H
