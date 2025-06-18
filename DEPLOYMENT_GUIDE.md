# FOMC Statement Classifier - Complete Deployment Guide

**Author:** Manus AI  
**Date:** June 2025  
**Version:** 2.0 (Streamlit Single App)

## Executive Summary

This comprehensive deployment guide provides detailed instructions for deploying the FOMC Statement Classifier web application, which utilizes fine-tuned FinBERT models to classify Federal Open Market Committee statements across six key financial attributes. The application transforms your existing Gradio-based prototype into a single, self-contained Streamlit web application with both interactive classification capabilities and historical data browsing functionality.

This updated architecture consolidates all logic into a single Streamlit application, simplifying deployment and management. This guide covers local testing, cloud deployment options, and production considerations for making your research accessible to financial institutions, market analysts, and macroeconomic researchers.

## Table of Contents

1. [Project Overview and Architecture](#project-overview)
2. [Prerequisites and Setup Requirements](#prerequisites)
3. [Local Development and Testing](#local-development)
4. [Cloud Deployment Strategies](#cloud-deployment)
5. [Production Considerations](#production)
6. [Troubleshooting and Maintenance](#troubleshooting)
7. [Performance Optimization](#performance)
8. [Security and Compliance](#security)

## Project Overview and Architecture {#project-overview}

### Application Architecture

The FOMC Statement Classifier now operates as a single, self-contained **Streamlit Application**. This architecture simplifies deployment by consolidating all components—user interface, machine learning model inference, and data handling—into one Python application. This approach is ideal for rapid prototyping, internal tools, and applications where the primary goal is ease of deployment and interactive data exploration.

The Streamlit application provides an intuitive, interactive web interface that closely mirrors the functionality of your original Gradio application while offering enhanced user experience and professional presentation. It features dedicated sections for text input, historical data browsing, and results visualization. Users can either input custom FOMC statement text or select from historical statements organized by year and month. The interface provides real-time classification results with confidence scores and color-coded confidence indicators, making it easy for users to assess the reliability of predictions.

All machine learning model inference operations are now performed directly within the Streamlit application. This includes loading your fine-tuned FinBERT models for Sentiment, Economic Growth, Employment Growth, Inflation, Medium Term Rate, and Policy Rate attributes. Historical data access is also managed directly within the Streamlit app, with the application loading your Excel dataset into an in-memory pandas DataFrame during startup.

The **Docker Containerization** layer ensures consistent deployment across different environments, from local development machines to cloud platforms. The containerized approach eliminates dependency conflicts, simplifies deployment procedures, and provides isolation between the application and host system. The Docker configuration includes proper signal handling and optimized image layering for efficient builds and deployments.

### Data Flow and Processing Pipeline

In this consolidated architecture, the data flow is simplified. When a user submits text for classification, the Streamlit application directly calls the internal classification functions. These functions perform tokenization using the loaded tokenizers, process the input through the transformer architecture, and generate classification probabilities using softmax activation. The system then determines the most likely class for each attribute and calculates confidence scores. These results are then directly rendered within the Streamlit interface.

For historical data access, the application loads your Excel file into an in-memory pandas DataFrame during its initial startup. The Streamlit UI then directly queries this DataFrame to filter statements by year and month, providing users with the ability to explore historical FOMC communications and compare model predictions against actual labeled data when available.


## Prerequisites and Setup Requirements {#prerequisites}

### System Requirements

Before deploying the FOMC Statement Classifier, ensure your system meets the following minimum requirements. These specifications are designed to support the computational demands of running multiple transformer-based models simultaneously while maintaining responsive user interactions.

**Hardware Requirements:**
- **CPU:** Minimum 4 cores (8 cores recommended for production)
- **RAM:** 8GB minimum (16GB recommended, 32GB for high-traffic deployments)
- **Storage:** 10GB free space for application and model files
- **GPU:** Optional but recommended for faster inference (CUDA-compatible GPU with 4GB+ VRAM)

**Software Requirements:**
- **Operating System:** Linux (Ubuntu 20.04+ recommended), macOS 10.15+, or Windows 10+
- **Docker:** Version 20.10 or later with Docker Compose (for containerized deployment)
- **Python:** 3.9+ (if running without Docker)
- **Git:** For version control and deployment workflows

### Model and Data Preparation

The successful deployment of your FOMC classifier depends on properly organizing your historical data. Your fine-tuned models will be loaded directly from Hugging Face, simplifying the deployment process.

**Historical Data Preparation:**
Your Excel file containing historical FOMC statements and their labeled attributes should be placed in the `data/` directory and named `Sentiment&Attributes_Classification.xlsx`. The file should maintain the same structure as used in your Gradio application, with columns for Date, statement_content, and the six attribute labels. The application expects dates in YYYYMMDD format and will automatically parse them for the historical browsing functionality.

### Environment Configuration

The application uses environment variables to configure various aspects of its operation, providing flexibility for different deployment scenarios. Understanding these configuration options is crucial for successful deployment and ongoing maintenance.

**Core Environment Variables:**
- `EXCEL_PATH`: Path to your historical data Excel file (default: `./data/Sentiment&Attributes_Classification.xlsx`)
- `CUDA_VISIBLE_DEVICES`: GPU device selection for CUDA-enabled deployments
- `PYTHONPATH`: Python path configuration for module imports

**Deployment-Specific Variables:**
Different deployment platforms may require additional environment variables. For Hugging Face Spaces, the `SPACE_ID` variable is automatically set and used by the application to detect the deployment environment. For cloud platforms like Render or Railway, platform-specific variables help the application adapt its configuration accordingly.

### Security Considerations

When preparing for deployment, especially for public access, several security considerations must be addressed to protect both your application and users' data. The application processes user-submitted text through your machine learning models but does not store or log this data by default. However, ensure that your deployment environment complies with relevant data protection regulations, particularly if serving users in jurisdictions with strict privacy requirements such as GDPR or CCPA.

**Model Protection:**
Your fine-tuned models, loaded from Hugging Face, represent significant intellectual property and research investment. While Hugging Face provides robust hosting, consider additional measures such as private repositories or access controls for highly sensitive models.


## Local Development and Testing {#local-development}

### Initial Setup and Configuration

Local development and testing provide the foundation for successful deployment by allowing you to verify that all components function correctly before moving to production environments. This section provides comprehensive instructions for setting up the FOMC classifier on your local machine, testing its functionality, and troubleshooting common issues.

**Step 1: Project Setup**
Begin by organizing your project files according to the structure created by the deployment scripts. Create a new directory for your project and copy the generated application files:

```bash
mkdir fomc_classifier_deployment
cd fomc_classifier_deployment
```

Copy all the generated files from the `/home/ubuntu/fomc_classifier/` directory to your project directory. This includes the `app.py` (your Streamlit application), Docker configuration files, and deployment scripts.

**Step 2: Historical Data Integration**
Place your Excel file containing historical FOMC statements in the `data/` directory. The file should be named `Sentiment&Attributes_Classification.xlsx` to match the default configuration. If you need to use a different filename or location, update the `EXCEL_PATH` environment variable accordingly.

Verify that your Excel file contains the expected columns and data format. The application expects a Date column in YYYYMMDD format, a statement_content column with the full text of FOMC statements, and columns for each of the six classification attributes with their corresponding labels.

### Docker-Based Local Testing

Docker provides the most reliable method for local testing as it closely mirrors the production deployment environment. This approach eliminates potential issues related to Python version differences, dependency conflicts, or operating system variations.

**Building the Docker Image**
Navigate to your project directory and build the Docker image using the provided Dockerfile:

```bash
docker build -t fomc-classifier .
```

This command creates a Docker image containing your application, models, and all dependencies. The build process may take several minutes, particularly when installing PyTorch and other machine learning libraries. Monitor the build output for any errors or warnings that might indicate configuration issues.

**Running the Application**
Once the image is built successfully, run the container using Docker Compose for the most straightforward setup:

```bash
docker-compose up
```

This command starts the Streamlit application within the container, with appropriate port mappings to make the application accessible from your local machine. The application will be available at `http://localhost:8501`.

**Alternative Docker Run Command**
If you prefer not to use Docker Compose, you can run the container directly:

```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  fomc-classifier
```

This approach provides more explicit control over volume mounting and port mapping, which can be useful for debugging or customization.

### Native Python Testing

For development and debugging purposes, you may prefer to run the application directly using Python rather than Docker. This approach provides easier access to logs, debugging tools, and code modification workflows.

**Setup and Testing**
First, set up a Python virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Configure the environment variable for local testing:

```bash
export EXCEL_PATH="./data/Sentiment&Attributes_Classification.xlsx"
```

Start the Streamlit application:

```bash
streamlit run app.py --server.port 8501
```

The Streamlit application will automatically open in your default web browser, or you can navigate to `http://localhost:8501` manually.

### Functional Testing and Validation

Comprehensive testing ensures that your application functions correctly across all intended use cases. This section outlines systematic testing procedures to validate the application.

**Application Testing**
Test the classification functionality by entering sample text and verifying that results are displayed correctly with appropriate formatting and confidence scores.

Test the historical data browsing functionality. Verify that you can select years, months, and individual statements, and that the selected statement content loads into the input text area. Also, check if the 

