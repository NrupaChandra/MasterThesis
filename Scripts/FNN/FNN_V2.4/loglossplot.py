#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt

# -----------------------------
# Set your log file and output file names here
# -----------------------------
LOG_FILE = "Test.out.47543174.txt"      
OUTPUT_FILE = "loss_plot.png"      

def parse_log_file(log_file):
    """
    Parse the log file to extract epochs, training losses, and validation losses.
    """
    epochs = []
    training_losses = []
    validation_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Look for an epoch header line (e.g., "Epoch 1/1000 Summary:")
            epoch_match = re.match(r"Epoch\s+(\d+)/", line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                epochs.append(epoch)
            
            # Look for the training loss line.
            train_match = re.search(r"Training Loss:\s*([\d\.eE+-]+)", line)
            if train_match:
                training_losses.append(float(train_match.group(1)))
            
            # Look for the validation loss line.
            valid_match = re.search(r"Validation Loss:\s*([\d\.eE+-]+)", line)
            if valid_match:
                validation_losses.append(float(valid_match.group(1)))
                
    return epochs, training_losses, validation_losses

def plot_losses(epochs, training_losses, validation_losses, output_file):
    """
    Create a plot of training and validation loss vs epoch with a logarithmic y-axis and save it.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, marker='o', label='Training Loss')
    plt.plot(epochs, validation_losses, marker='o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epoch (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")

if __name__ == '__main__':
    epochs, training_losses, validation_losses = parse_log_file(LOG_FILE)
    
    if not (len(epochs) == len(training_losses) == len(validation_losses)):
        print("Warning: The number of epochs, training losses, and validation losses do not match exactly.")
    
    plot_losses(epochs, training_losses, validation_losses, OUTPUT_FILE)
