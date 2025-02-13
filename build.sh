#!/bin/bash

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies from requirements.txt
pip install --upgrade pip  # Ensure you're using the latest pip
pip install -r requirements.txt

# Download the spaCy model (optional if you want to do it during build)
python -m spacy download en_core_web_sm

# You can add other setup tasks here, like migrations, if needed
