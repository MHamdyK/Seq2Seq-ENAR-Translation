# src/model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate

def build_baseline_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
    """
    Build a standard seq2seq model without attention.
    """
    # Encoder
    encoder_inputs = Input(shape=(None, num_encoder_tokens), name="encoder_inputs")
    encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
    _, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(None, num_decoder_tokens), name="decoder_inputs")
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="baseline_seq2seq")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_attention_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
    """
    Build a seq2seq model with attention mechanism.
    """
    # Encoder
    encoder_inputs = Input(shape=(None, num_encoder_tokens), name="encoder_inputs")
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(None, num_decoder_tokens), name="decoder_inputs")
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    
    # Attention mechanism
    attention_layer = Attention(name="attention_layer")
    attention_output = attention_layer([decoder_outputs, encoder_outputs])
    
    # Concatenate attention output with decoder LSTM output
    decoder_combined_context = Concatenate(axis=-1, name="concat_layer")([decoder_outputs, attention_output])
    
    # Final output layer
    decoder_dense = Dense(num_decoder_tokens, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_combined_context)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="attention_seq2seq")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
