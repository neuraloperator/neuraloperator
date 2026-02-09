import matplotlib.pyplot as plt
import os
import torch
import pandas as pd

def plot_history( save_path=None):
    """Plot the training and validation loss history.

    Args:
        loss_history (dict): Dictionary with keys 'train' and 'val' containing lists of loss values.
        save_path (str, optional): Path to save the plot image. If None, the plot is shown. 

        columns in CSV:
        epoch, time, train_err, enc_u_loss, enc_v_loss, enc_cp_loss, enc_lognutratio_loss, lr

    """ 
    #with open(os.path.join(save_path, 'loss_history.csv'), 'r') as f:
    loss_history = pd.read_csv(os.path.join(save_path, 'loss_history.csv'))
    
    # Strip whitespace from column names
    loss_history.columns = loss_history.columns.str.strip()
    
    print (loss_history.head())
    for col in loss_history.columns:
        print(f"Column: '{col}'")
    weights = [0.6, 1.0, 0.8, 0.1]
    loss_history['avg_loss'] = (weights[0] * loss_history['enc_u_loss'] + 
                                weights[1] * loss_history['enc_v_loss'] + 
                                weights[2] * loss_history['enc_cp_loss'] + 
                                weights[3] * loss_history['enc_lognutratio_loss']) / sum(weights)  
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history['epoch'], loss_history['train_err'], label='Training Error', color='black', linewidth=2)
    plt.plot(loss_history['epoch'], loss_history['enc_u_loss'], label='Encoded U Loss')
    plt.plot(loss_history['epoch'], loss_history['enc_u_loss'], label='Encoded U Loss')
    plt.plot(loss_history['epoch'], loss_history['enc_v_loss'], label='Encoded V Loss')
    plt.plot(loss_history['epoch'], loss_history['enc_cp_loss'], label='Encoded Cp Loss')
    plt.plot(loss_history['epoch'], loss_history['enc_lognutratio_loss'], label='Encoded log(nut/nu) Loss')
    plt.plot(loss_history['epoch'], loss_history['lr'], label='Learning Rate')
    plt.plot(loss_history['epoch'], loss_history['avg_loss'], label='Average Encoded Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)      
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    history_path = "/home/timm/Projects/PIML/neuraloperator/tims/airfransX5Y4/checkpoints"
    plot_history(save_path=history_path)