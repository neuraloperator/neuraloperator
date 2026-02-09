#!/bin/bash

# AirFrans Model Viewer Launcher
# This script launches the Streamlit app for viewing AirFrans models and datasets

echo "🌊 Starting AirFrans Model Viewer..."

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Navigate to the app directory
cd "$SCRIPT_DIR"

# Check if streamlit is available
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed. Please install it first:"
    echo "   pip install streamlit"
    exit 1
fi

# Launch the Streamlit app
echo "🚀 Launching Streamlit app..."
echo "📂 App directory: $SCRIPT_DIR"
echo "🌐 The app will open in your default web browser"
echo ""

streamlit run airfrans_model_viewer.py