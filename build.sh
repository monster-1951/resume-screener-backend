#!/bin/bash

# Remove any existing virtual environment
rm -rf venv

# Create a new virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Ensure spaCy model is downloaded (if needed)
python -m spacy download en_core_web_sm

# Run the application using the virtual environment
exec venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
