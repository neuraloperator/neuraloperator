#!/usr/bin/env python3
"""
Quick script to create animated GIF from field prediction PNGs
"""

import os
import sys
import subprocess

# Add the current directory to Python path
sys.path.append('/home/timm/Projects/PIML/neuraloperator')

# Import the GIF creator
from create_gif_from_pngs import create_animated_gif

def main():
    input_dir = "/home/timm/Projects/PIML/neuraloperator/tims/airfransX5Y4/checkpoints/field"
    output_file = "field_predictions_animation.gif"
    
    print("🎬 Creating animated GIF from field prediction images...")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output file: {output_file}")
    
    if not os.path.exists(input_dir):
        print(f"❌ Directory does not exist: {input_dir}")
        print("Make sure you have run some training epochs to generate field prediction images.")
        return
    
    # Create the animated GIF
    create_animated_gif(
        input_dir=input_dir,
        output_path=output_file,
        duration=800,  # 0.8 seconds per frame
        loop=0         # Infinite loop
    )

if __name__ == "__main__":
    main()