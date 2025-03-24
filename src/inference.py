# src/inference.py

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

def build_inference_models(baseline_model, latent_dim, num_encoder_tokens, num_decoder_tokens):
    """
    Create inference encoder and decoder models from a trained baseline seq2seq model.
    """
    # Encoder inference model
    encoder_inputs = baseline_model.input[0]
    encoder_lstm = baseline_model.get_layer("encoder_lstm")
    _, state_h_enc, state_c_enc = encoder_lstm(encoder_inputs)
    encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])
    
    # Decoder inference model
    decoder_inputs = baseline_model.input[1]
    decoder_state_input_h = Input(shape=(latent_dim,), name="input_h")
    decoder_state_input_c = Input(shape=(latent_dim,), name="input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_lstm = baseline_model.get_layer("decoder_lstm")
    decoder_dense = baseline_model.get_layer("decoder_dense")
    
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs, state_h, state_c])
    
    return encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model, output_token_index, reverse_output_char_index, num_decoder_tokens, max_decoder_seq_length):
    """
    Decode an input sequence to generate the translated output.
    """
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    
    # Create target sequence with the start token.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, output_token_index["<start>"]] = 1.0
    
    decoded_sentence = ""
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_output_char_index.get(sampled_token_index, "")
        
        if sampled_char == "<end>" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        else:
            decoded_sentence += sampled_char
        
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        
        states_value = [h, c]
    
    return decoded_sentence
