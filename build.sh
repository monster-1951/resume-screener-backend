#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e  

# Install Python dependencies
pip install -r requirements.txt  

# Download the compatible spaCy model manually
python -m spacy download en-core-web-sm --direct  

# Verify installation (optional)
python -c "import spacy; spacy.load('en-core-web-sm')"
