#!/bin/bash

# Start Streamlit frontend in the foreground
echo "Starting Streamlit application..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true


