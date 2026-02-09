#!/usr/bin/env python3

# Quick test to run actual training for just 1 step to see debug output
import sys
sys.path.append('/home/timm/Projects/PIML/neuraloperator')

import os
os.chdir('/home/timm/Projects/PIML/neuraloperator')

# Run the existing training script but limit it to 1 epoch
os.system("timeout 40s python tims/airfransX5Y4/start_train_airfrans_with_wandb.py || echo 'Training stopped after 40 seconds'")