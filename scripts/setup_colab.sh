#!/bin/bash

echo "Installing Libraries & Project..."
pip install -r requirements.txt

echo "Checking Data Connection..."

if [ -d "/content/drive/MyDrive/sapienza_fds_fruit_classification/data" ]; then
    echo "SUCCESS: Data folder found on Google Drive."
else
    echo "ERROR: Data folder not found. Make sure you mounted Drive!"
fi

echo ">>> Setup Complete!"
