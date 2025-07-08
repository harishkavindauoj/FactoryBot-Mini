"""Improved model architecture with better performance."""
import tensorflow as tf
from keras import Model
from tensorflow.keras.layers import (
    Input, GRU, LSTM, Conv1D, GlobalMaxPooling1D, Dense,
    Dropout, BatchNormalization, Concatenate, Attention,
    LayerNormalization, MultiHeadAttention
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np


def build_improved_model(config):
    """Build an improved multi-task model with better architecture."""
    input_shape = tuple(config['model']['input_shape'])
    hidden_units = config['model']['hidden_units']
    dropout_rate = config['model']['dropout']

    # Input layer
    inp = Input(shape=input_shape, name='sensor_input')

    # Multi-scale feature extraction
    # Path 1: Short-term patterns (CNN)
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inp)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(dropout_rate)(conv1)

    conv2 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(inp)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(dropout_rate)(conv2)

    # Path 2: Long-term dependencies (Bidirectional GRU)
    gru1 = GRU(hidden_units, return_sequences=True, dropout=dropout_rate,
               recurrent_dropout=dropout_rate)(inp)
    gru1 = LayerNormalization()(gru1)

    gru2 = GRU(hidden_units // 2, return_sequences=True, dropout=dropout_rate,
               recurrent_dropout=dropout_rate)(gru1)
    gru2 = LayerNormalization()(gru2)

    # Attention mechanism for important time steps
    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=hidden_units // 4
    )(gru2, gru2)
    attention_output = LayerNormalization()(attention_output)

    # Global pooling for different paths
    conv_pool1 = GlobalMaxPooling1D()(conv1)
    conv_pool2 = GlobalMaxPooling1D()(conv2)
    gru_pool = GlobalMaxPooling1D()(attention_output)

    # Combine all features
    combined = Concatenate()([conv_pool1, conv_pool2, gru_pool])

    # Shared dense layers
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(combined)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Task-specific heads
    # Risk classification head
    risk_branch = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    risk_branch = Dropout(dropout_rate)(risk_branch)
    risk_output = Dense(4, activation='softmax', name='risk_level')(risk_branch)

    # TTF regression head
    ttf_branch = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    ttf_branch = Dropout(dropout_rate)(ttf_branch)
    ttf_output = Dense(1, activation='linear', name='ttf')(ttf_branch)

    # Create model
    model = Model(inputs=inp, outputs={
        "risk_level": risk_output,
        "ttf": ttf_output
    })

    # Improved compilation with better loss weighting
    model.compile(
        optimizer=Adam(learning_rate=config['training']['initial_lr']),
        loss={
            'risk_level': 'categorical_crossentropy',
            'ttf': 'huber'  # More robust than MSE for regression
        },
        loss_weights={
            'risk_level': 0.3,  # Lower weight for classification
            'ttf': 0.7  # Higher weight for regression (main task)
        },
        metrics={
            'risk_level': ['accuracy'],
            'ttf': ['mae', 'mse']
        }
    )

    return model


def build_transformer_model(config):
    """Alternative: Transformer-based model for time series."""
    input_shape = tuple(config['model']['input_shape'])
    d_model = config['model']['hidden_units']

    inp = Input(shape=input_shape)

    # Positional encoding (simple version)
    x = Dense(d_model)(inp)

    # Multi-head self-attention
    attention = MultiHeadAttention(
        num_heads=8, key_dim=d_model // 8
    )(x, x)
    attention = LayerNormalization()(attention + x)

    # Feed-forward network
    ffn = Dense(d_model * 2, activation='relu')(attention)
    ffn = Dense(d_model)(ffn)
    ffn = LayerNormalization()(ffn + attention)

    # Global pooling
    pooled = GlobalMaxPooling1D()(ffn)

    # Dense layers
    x = Dense(128, activation='relu')(pooled)
    x = Dropout(0.3)(x)

    # Outputs
    risk_output = Dense(4, activation='softmax', name="risk_level")(x)
    ttf_output = Dense(1, activation='linear', name="ttf")(x)
    model = Model(inputs=inp, outputs={
        "risk_level": risk_output,
        "ttf": ttf_output
    })

    model.compile(
        optimizer=Adam(learning_rate=config['training']['initial_lr']),
        loss={'risk_level': 'categorical_crossentropy', 'ttf': 'huber'},
        loss_weights={'risk_level': 0.3, 'ttf': 0.7},
        metrics={'risk_level': ['accuracy'], 'ttf': ['mae']}
    )

    return model


def build_ensemble_model(config):
    """Build an ensemble of different architectures."""
    input_shape = tuple(config['model']['input_shape'])

    # Shared input
    inp = Input(shape=input_shape)

    # Model 1: CNN-based
    cnn_branch = Conv1D(64, 3, activation='relu')(inp)
    cnn_branch = Conv1D(64, 3, activation='relu')(cnn_branch)
    cnn_branch = GlobalMaxPooling1D()(cnn_branch)
    cnn_branch = Dense(64, activation='relu')(cnn_branch)

    # Model 2: GRU-based
    gru_branch = GRU(64, return_sequences=True)(inp)
    gru_branch = GRU(32)(gru_branch)
    gru_branch = Dense(64, activation='relu')(gru_branch)

    # Model 3: LSTM-based
    lstm_branch = LSTM(64, return_sequences=True)(inp)
    lstm_branch = LSTM(32)(lstm_branch)
    lstm_branch = Dense(64, activation='relu')(lstm_branch)

    # Combine branches
    combined = Concatenate()([cnn_branch, gru_branch, lstm_branch])

    # Final layers
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.3)(x)

    risk_output = Dense(4, activation='softmax', name='risk_level')(x)
    ttf_output = Dense(1, activation='linear', name='ttf')(x)

    model = Model(inputs=inp, outputs={
        "risk_level": risk_output,
        "ttf": ttf_output
    })

    model.compile(
        optimizer=Adam(learning_rate=config['training']['initial_lr']),
        loss={'risk_level': 'categorical_crossentropy', 'ttf': 'huber'},
        loss_weights={'risk_level': 0.3, 'ttf': 0.7},
        metrics={'risk_level': ['accuracy'], 'ttf': ['mae']}
    )

    return model


# Updated build_model function
def build_model(config):
    """Main model builder with architecture selection."""
    architecture = config['model'].get('architecture', 'improved')

    if architecture == 'improved':
        return build_improved_model(config)
    elif architecture == 'transformer':
        return build_transformer_model(config)
    elif architecture == 'ensemble':
        return build_ensemble_model(config)
    else:
        # Fallback to original simple model
        return build_simple_model(config)


def build_simple_model(config):
    """Original simple model for comparison."""
    input_shape = tuple(config['model']['input_shape'])
    model_type = config['model']['type']
    hidden_units = config['model']['hidden_units']
    dropout_rate = config['model']['dropout']

    inp = Input(shape=input_shape)

    if model_type == "GRU":
        x = GRU(hidden_units, return_sequences=False)(inp)
    elif model_type == "CNN":
        x = Conv1D(filters=hidden_units, kernel_size=3, activation='relu')(inp)
        x = GlobalMaxPooling1D()(x)
    else:
        raise ValueError("Unknown model type")

    x = Dropout(dropout_rate)(x)

    risk_output = Dense(4, activation="softmax", name="risk_level")(x)
    ttf_output = Dense(1, activation="linear", name="ttf")(x)

    model = Model(inputs=inp, outputs={
        "risk_level": risk_output,
        "ttf": ttf_output
    })

    model.compile(
        loss={"risk_level": "categorical_crossentropy", "ttf": "mse"},
        optimizer="adam",
        metrics={"risk_level": "accuracy", "ttf": "mae"},
    )
    return model