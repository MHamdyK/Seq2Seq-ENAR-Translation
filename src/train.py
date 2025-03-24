# src/train.py

from tensorflow.keras.callbacks import EarlyStopping

def train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, 
                batch_size, epochs, validation_split, patience=5):
    """
    Train the provided model using EarlyStopping.
    """
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split,
                        callbacks=[early_stop])
    return history
