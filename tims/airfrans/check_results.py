#!/usr/bin/env python
"""
Quick script to check for existing results and create visualizations.
"""

import torch
import subprocess
from pathlib import Path

def main():
    """Check for existing results and offer visualization options."""
    
    print("=== AirFRANS Training Results Checker ===\n")
    
    # Check for local checkpoints
    checkpoint_dir = Path("./checkpoints")
    local_checkpoints = list(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
    
    print(f"Local checkpoints found: {len(local_checkpoints)}")
    for ckpt in local_checkpoints:
        print(f"  - {ckpt.name} ({ckpt.stat().st_size / (1024*1024):.1f} MB)")
    
    # Check if WandB has any runs
    try:
        import wandb
        # Initialize wandb
        api = wandb.Api()
        runs = list(api.runs(f"tim-mak-ntnu/neuraloperator-airfrans"))
        print(f"\nWandB runs found: {len(runs)}")
        
        if runs:
            print("Recent runs:")
            for run in runs[:3]:  # Show last 3 runs
                status = run.state
                created = run.created_at.strftime("%Y-%m-%d %H:%M")
                print(f"  - {run.name} ({status}) - {created}")
                
                # Check if run has artifacts
                if hasattr(run, 'logged_artifacts') and run.logged_artifacts():
                    print(f"    └─ Has model artifacts")
        
    except Exception as e:
        print(f"\nCould not check WandB runs: {e}")
    
    # Offer visualization options
    print("\n" + "="*50)
    print("VISUALIZATION OPTIONS:")
    print("="*50)
    
    if local_checkpoints:
        latest_checkpoint = max(local_checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"\n1. Visualize with local checkpoint:")
        print(f"   python plot_results.py --checkpoint {latest_checkpoint}")
        
        print(f"\n2. Evaluate local checkpoint:")
        print(f"   python evaluate_model.py --checkpoint {latest_checkpoint}")
    
    print(f"\n3. Train new model with WandB:")
    print(f"   python train_with_wandb.py")
    
    if 'runs' in locals() and runs:
        print(f"\n4. Download and visualize from WandB:")
        latest_run = runs[0]
        print(f"   python plot_results.py --wandb_run {latest_run.id}")
    
    print(f"\n5. Quick test without checkpoint (random weights):")
    print(f"   python plot_results.py --checkpoint none")


if __name__ == "__main__":
    main()