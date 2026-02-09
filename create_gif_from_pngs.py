#!/usr/bin/env python3
"""
Script to create an animated GIF from PNG files in a directory.
Usage: python create_gif_from_pngs.py [--duration DURATION] [--output OUTPUT_NAME]
"""

import os
import glob
import re
from PIL import Image
import argparse
from pathlib import Path

def natural_sort_key(text):
    """
    Sort key for natural ordering of filenames with numbers.
    E.g., 'file_1.png', 'file_2.png', ..., 'file_10.png'
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def create_animated_gif(input_dir, output_path, duration=500, loop=0):
    """
    Create an animated GIF from PNG files in a directory.
    
    Args:
        input_dir (str): Directory containing PNG files
        output_path (str): Output path for the GIF file
        duration (int): Duration between frames in milliseconds
        loop (int): Number of loops (0 = infinite)
    """
    
    # Find all PNG files
    png_pattern = os.path.join(input_dir, "*.png")
    png_files = glob.glob(png_pattern)
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    # Sort files naturally (handles numbers correctly)
    png_files.sort(key=natural_sort_key)
    
    print(f"Found {len(png_files)} PNG files")
    print(f"First file: {os.path.basename(png_files[0])}")
    print(f"Last file: {os.path.basename(png_files[-1])}")
    
    # Load images
    images = []
    for png_file in png_files:
        try:
            img = Image.open(png_file)
            # Convert to RGB if necessary (GIF doesn't support RGBA)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
            print(f"Loaded: {os.path.basename(png_file)} ({img.size})")
        except Exception as e:
            print(f"Error loading {png_file}: {e}")
            continue
    
    if not images:
        print("No valid images loaded")
        return
    
    # Create animated GIF
    print(f"Creating animated GIF with {len(images)} frames...")
    print(f"Duration per frame: {duration}ms")
    print(f"Output: {output_path}")
    
    # Save as animated GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    
    print(f"✅ Animated GIF created successfully: {output_path}")
    
    # Print file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="Create animated GIF from PNG files")
    parser.add_argument(
        "--input_dir", 
        default="/home/timm/Projects/PIML/neuraloperator/tims/airfransX5Y4/checkpoints/field",
        help="Input directory containing PNG files"
    )
    parser.add_argument(
        "--output", 
        default="field_animation.gif",
        help="Output GIF filename"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=800,
        help="Duration between frames in milliseconds (default: 800)"
    )
    parser.add_argument(
        "--loop", 
        type=int, 
        default=0,
        help="Number of loops (0 = infinite, default: 0)"
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory does not exist: {args.input_dir}")
        return
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🎬 Creating animated GIF from PNG files")
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output}")
    print("-" * 50)
    
    create_animated_gif(
        input_dir=args.input_dir,
        output_path=args.output,
        duration=args.duration,
        loop=args.loop
    )

if __name__ == "__main__":
    main()