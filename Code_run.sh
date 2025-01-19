#!/bin/bash

cd Codes || { echo "Codes directory not found!"; exit 1; }

cd dynamic_sign || { echo "dynamic_sign directory not found!"; exit 1; }

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
    exit 1
fi

# Run the test-app.py
if [ -f "test-app.py" ]; then
    echo "Running test-app.py..."
    python test-app.py
else
    echo "test-app.py not found!"
    exit 1
fi
