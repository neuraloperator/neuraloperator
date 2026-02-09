import matplotlib.pyplot as plt
import os
import torch
import pandas as pd

def plot_cl_cd( save_path=None):
    """Plot the training and validation loss history.

    Args:
        train or test history (csv)
        save_path (str, optional): Path to save the plot image. If None, the plot is shown. 

        columns in CSV:
        epoch, time, train_err, enc_u_loss, enc_v_loss, enc_cp_loss, enc_lognutratio_loss, lr

    """ 
    #with open(os.path.join(save_path, 'loss_history.csv'), 'r') as f:
    results = pd.read_csv(os.path.join(save_path))
    
    # Strip whitespace from column names
    results.columns = results.columns.str.strip()

    print (results.head())
    for col in results.columns:
        print(f"Column: '{col}'")





    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.scatter(results['truth_cdp'], results['cdp'], marker='o', label='Truth_Cdp')
    plt.plot([ 0.0, 0.04], [0.0, 0.04], 'r--', label='Identity')  # y=x reference line
    plt.grid(True)      

    plt.title('Cdp vs Truth Cdp and Clp vs Truth Clp')
    plt.xlabel('Truth Cdp')
    plt.ylabel('Prediction Cdp')    

    plt.subplot(2, 1, 2)
    plt.scatter(results['truth_clp'], results['clp'], marker='o', label='Truth Clp')
    plt.plot([-0.5, 1.5], [-0.5, 1.5], 'r--', label='Identity')  # y=x reference line
    plt.xlabel('Truth Clp')
    plt.ylabel('Prediction Clp')
    plt.legend()
    plt.grid(True)      
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(results['aoa_deg'], results['clp'], marker='o', label='Clp')
    plt.scatter(results['aoa_deg'], results['truth_clp'], marker='o', label='Truth Clp')
    plt.grid(True)      
    plt.title('Clp and Truth Cp vs aoa')
    plt.xlabel('Truth Cdp')
    plt.ylabel('Prediction Cdp')    
    plt.tight_layout()
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # Example usage
    history_path = "/home/timm/Projects/PIML/neuraloperator/tims/airfransX5Y4/test_model/plots_512_full_test/airfoil_512_full_test_results.csv"
    plot_cl_cd(save_path=history_path)