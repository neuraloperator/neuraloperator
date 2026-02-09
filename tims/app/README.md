# AirFrans Model Viewer

A Streamlit web application for loading and visualizing trained AirFrans models and datasets.

## Features

🔧 **Model Loading**
- Load saved models from directory containing:
  - `data_processor.pt` - Data preprocessing/postprocessing pipeline
  - `best_model_state_dict.pt` - Trained model weights
  - `best_model_metadata.pkl` or `best_model_metadata.pt` - Training metadata and configuration
- Display model training information (epoch, losses, learning rate)

📊 **Dataset Exploration**
- Load AirFrans X5Y4 dataset with multiple splits
- Available splits from manifest.json:
  - `full_train` - Full training set
  - `aoa_train` - Angle of attack variations
  - `reynolds_train` - Reynolds number variations
  - `scarce_train` - Limited training data
  - `full_test`, `aoa_test`, `reynolds_test` - Test sets
- Configure train/test splits dynamically
- View dataset statistics and sample counts

🎨 **Sample Visualization**
- Interactive sample plotting from any loaded dataset split
- Visualize input fields:
  - Flow mask (interior/exterior regions)
  - Signed distance field (SDF)
  - Flow parameters (U_inf, V_inf, Reynolds number)
- Visualize output fields:
  - U-Deficit (velocity u-component deficit)
  - V-Deficit (velocity v-component deficit) 
  - Cp (pressure coefficient)
  - log(nut_ratio) (turbulent viscosity ratio)
- Shared velocity scaling for consistent comparison
- Sample index selection with bounds checking

## Usage

### Method 1: Using the launcher script
```bash
cd /home/timm/Projects/PIML/neuraloperator/tims/app
./run_app.sh
```

### Method 2: Direct streamlit command
```bash
cd /home/timm/Projects/PIML/neuraloperator/tims/app
streamlit run airfrans_model_viewer.py
```

### Method 3: From Python environment
```python
import streamlit as st
import subprocess
subprocess.run(["streamlit", "run", "/path/to/app/airfrans_model_viewer.py"])
```

## Interface Guide

### Left Panel: Configuration
1. **Model Loading**
   - Enter path to model directory
   - Click "Load Model Files" to load saved model
   - View model metadata (epoch, losses, etc.)

2. **Dataset Configuration**
   - Select one or more dataset splits
   - First split becomes training set
   - Additional splits become test sets
   - Click "Load Dataset" to initialize data loaders

### Right Panel: Visualization
1. **Data Loader Selection**
   - Choose between training set and test sets
   - Shows split names from your selection

2. **Sample Visualization**
   - Select sample index (0 to dataset_size-1)
   - Click "Plot Sample" to generate visualization
   - View input fields (top row) and output fields (bottom row)

## Dataset Path Configuration

The app is configured to use:
- **Dataset Path**: `/home/timm/Projects/PIML/Dataset_PT_FNO_X5Y4/TrainingX5Y4_consolidated`
- **Manifest File**: `manifest.json` (contains split definitions)

To use a different dataset path, modify the `DATASET_PATH` constant in `airfrans_model_viewer.py`.

## Requirements

- Python 3.8+
- Streamlit
- PyTorch
- Matplotlib
- NumPy
- Pandas
- The neuraloperator project dependencies

Install requirements:
```bash
pip install -r requirements.txt
```

## Troubleshooting

**Import Error**: If you get import errors for the neuraloperator modules, ensure:
1. You're running from the correct directory
2. The neuraloperator project is in your Python path
3. All dependencies are installed

**Dataset Not Found**: Verify the dataset path and manifest.json file exist:
```bash
ls /home/timm/Projects/PIML/Dataset_PT_FNO_X5Y4/TrainingX5Y4_consolidated/manifest.json
```

**Model Loading Issues**: Ensure your model directory contains all three required files:
- `data_processor.pt`
- `best_model_state_dict.pt` 
- `best_model_metadata.pkl` # is actually a torch file 

## Tips

- Start with loading the dataset first to explore available data
- Use the training set for quick exploration (usually larger sample count)
- The shared velocity scaling helps compare u/v deficit fields
- Sample indices correspond to the order in the dataset split
- Model loading is optional - you can explore datasets without a trained model