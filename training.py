import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import keras_tuner as kt
import streamlit as st
import numpy as np

# --- CUSTOM CALLBACK FOR REAL-TIME PLOTTING ---
class StreamlitPlotCallback(Callback):
    def __init__(self, plot_container):
        super().__init__()
        self.plot_container = plot_container
        self.losses = []
        self.val_losses = []
        self.accs = []
        self.val_accs = []
        
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accs.append(logs.get('accuracy'))
        self.val_accs.append(logs.get('val_accuracy'))
        
        # Update charts
        with self.plot_container:
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart({
                    "Training Loss": self.losses, 
                    "Validation Loss": self.val_losses
                }, height=250)
                st.caption("Loss over Epochs")
            with col2:
                st.line_chart({
                    "Training Accuracy": self.accs, 
                    "Validation Accuracy": self.val_accs
                }, height=250)
                st.caption("Accuracy over Epochs")

# --- MODEL BUILDERS ---

def build_manual_model(filters1, filters2, dense_units, dropout_rate, lr):
    model = Sequential([
        Conv2D(filters1, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(filters2, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_tuner_model(hp):
    model = Sequential()
    
    model.add(Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=96, step=32),
        kernel_size=3, activation='relu', input_shape=(32, 32, 1)
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    
    model.add(Conv2D(
        filters=hp.Int('conv_2_filter', min_value=64, max_value=128, step=32),
        kernel_size=3, activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
        activation='relu'
    ))
    
    model.add(Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model