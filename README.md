# Seq2Seq-ENAR-Translation
EN-AR translation using Seq2Seq LSTM RNN with TensorFlow using two approaches: a baseline model without attention and an enhanced model with an attention mechanism.

# Training Data google drive link:


## Directory Structure

```plaintext
Seq2Seq-ENAR-Translation/
├── data/
│   └── download_dataset.sh            # Shell file to download the dataset
├── src/
│   ├── __init__.py           # Package initializer
│   ├── config.py             # Configuration and hyperparameters
│   ├── preprocessing.py      # Data loading and preprocessing functions
│   ├── model.py              # Model building functions for baseline and attention models
│   ├── train.py              # Training routines
│   ├── inference.py          # Inference model builders and decode function
│   └── utils.py              # Utility functions (e.g., plotting)
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and instructions
└── .gitignore                # Files/folders to ignore in Git
```
## Setup and Usage

**1-clone the repo:**
```
git clone https://github.com/your_username/Seq2Seq-ENAR-Translation.git
cd Seq2Seq-ENAR-Translation
```

**2-Install Dependencies:**
```
pip install -r requirements.txt
```

**3-Data Preparation:**
Run the shell file in the `data` directory to download the dataset(it is a csv file that contians EN-AR pairs) then run the preprocessing module from a Python shell or notebook:
```
from src import load_and_preprocess_data
(encoder_input_data, decoder_input_data, decoder_target_data,
 input_texts, output_texts, input_token_index, output_token_index,
 max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens) = load_and_preprocess_data("/path/to/Dataset.txt", max_samples=10000)
```

**4-Training:**
to train the baseline model:
```
from src import build_baseline_model, train_model, LATENT_DIM, BATCH_SIZE, EPOCHS_BASELINE, VALIDATION_SPLIT
model = build_baseline_model(num_encoder_tokens, num_decoder_tokens, LATENT_DIM)
history_baseline = train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, 
                               batch_size=BATCH_SIZE, epochs=EPOCHS_BASELINE, validation_split=VALIDATION_SPLIT)
```

**5-Evaulation and Inference:**
use the `build_inference_models` and `decode_sequence` functions from `src/inference.py` to do the translation on new sentences

**6-Plotting**
you could either use the `plot_history` function from the `src/utils.py` to visualize the training curve, or create a plot diagram yourself.
