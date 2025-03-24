# src/utils.py

import matplotlib.pyplot as plt

def plot_history(histories, labels, title="Model Loss"):
    """
    Plot the training and validation loss curves for multiple training histories.
    """
    plt.figure(figsize=(10, 6))
    for history, label in zip(histories, labels):
        plt.plot(history.history['loss'], label=f"Train Loss ({label})")
        plt.plot(history.history['val_loss'], label=f"Val Loss ({label})")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
