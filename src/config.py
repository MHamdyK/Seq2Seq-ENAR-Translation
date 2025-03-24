# src/config.py

# Data parameters
DATA_FILE = "/content/Dataset.txt"  # Update this with the dataset I have in the google drive
MAX_SAMPLES = 10000

# Model and training parameters
LATENT_DIM = 256
BATCH_SIZE = 64
EPOCHS_BASELINE = 8
EPOCHS_ATTENTION = 5
VALIDATION_SPLIT = 0.2
