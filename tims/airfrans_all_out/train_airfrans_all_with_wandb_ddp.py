#!/usr/bin/env python
"""
Launch script for training FNO on AirFRANS dataset with WandB logging enabled.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run training with WandB enabled."""
    
    # Get the script directory
    script_dir = Path(__file__).parent
    training_script = script_dir / "train_airfrans_all_fno_weightedloss_ddp.py"
    
    # Use uv run python instead of direct venv path
    cmd = [
        "uv", "run", "torchrun",
        "--nproc_per_node=2",
        str(training_script),
        "--wandb.log", "True",
        "--wandb.project", "neuraloperator-airfrans",
        "--wandb.entity", "tim-mak-ntnu",  # Your WandB username
        "--wandb.name", "fno-airfrans-all-training-ddp",
        "--wandb.log_output", "False",  # Disable output logging to avoid shape issues
        "--opt.n_epochs", "1000",  # Short test run first
        "--data.batch_size", "16",  # per GPU batch size
        "--save_interval", "20",
        "--distributed.use_distributed", "True",
        "--verbose", "True"
    ]
    
    print("Starting FNO training with WandB logging...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the training
    result = subprocess.run(cmd, cwd="/home/timm/Projects/PIML/neuraloperator")
    
    if result.returncode == 0:
        print("Training completed successfully!")
        print("Check your WandB dashboard at: https://wandb.ai/tim-mak-ntnu/neuraloperator-airfrans")
    else:
        print("Training failed!")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()