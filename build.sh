#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e  

# Upgrade pip first (to avoid outdated package issues)
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt  

# Manually download and link the correct spaCy model
python -m spacy download en-core-web-sm  

# (Optional) Verify the model is installed correctly
python -c "import spacy; spacy.load('en-core-web-sm')"
