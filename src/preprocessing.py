# src/preprocessing.py

import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path, max_samples=10000):
    """
    Load the dataset from a file and preprocess it by adding start/end tokens,
    then one-hot encode the characters.
    """
    # Read the data (assumes tab-separated file with two columns: English, Arabic)
    data = pd.read_csv(file_path, sep="\t", header=None, names=["English", "Arabic"])
    data = data.head(max_samples)
    
    # Add start and end tokens
    data['English'] = "<start> " + data['English'] + " <end>"
    data['Arabic']  = "<start> " + data['Arabic']  + " <end>"
    
    # Convert text to lists
    input_texts = data['English'].tolist()
    output_texts = data['Arabic'].tolist()
    
    # Build character vocabularies
    input_characters = sorted(set(''.join(input_texts)))
    output_characters = sorted(set(''.join(output_texts)))
    
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(output_characters)
    
    max_encoder_seq_length = max(len(text) for text in input_texts)
    max_decoder_seq_length = max(len(text) for text in output_texts)
    
    input_token_index = {char: i for i, char in enumerate(input_characters)}
    output_token_index = {char: i for i, char in enumerate(output_characters)}
    
    # Initialize one-hot encoded data arrays
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    
    for i, (input_text, output_text) in enumerate(zip(input_texts, output_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        for t, char in enumerate(output_text):
            decoder_input_data[i, t, output_token_index[char]] = 1.0
            if t > 0:
                # Decoder target data is ahead of decoder input data by one timestep
                decoder_target_data[i, t - 1, output_token_index[char]] = 1.0
                
    return (encoder_input_data, decoder_input_data, decoder_target_data,
            input_texts, output_texts, input_token_index, output_token_index,
            max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens)
