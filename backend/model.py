# Stock Price Prediction Model
# Defines model architecture and prediction functions

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
from typing import Tuple, Optional, List, Dict
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import traceback
from datetime import datetime, timedelta
from tensorflow.keras import regularizers
from tensorflow.keras import layers
import joblib
import pandas as pd
from causal import prices_from_returns, estimated_weekday_dates

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SEQUENCE_LENGTH = 30  # Days of historical data used for each prediction
PREDICTION_DAYS = 5   # Number of days to predict ahead
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")  # Use absolute path


class AttentionLayer(layers.Layer):
    """Attention mechanism to focus on important time steps"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # inputs: shape (batch_size, time_steps, hidden_dim)
        attention_logits = self.attention(inputs)             # (batch_size, time_steps, 1)
        attention_weights = tf.nn.softmax(attention_logits, axis=1)  # across time_steps
        # Cast attention_weights to the same dtype as inputs for mixed precision compatibility
        attention_weights = tf.cast(attention_weights, inputs.dtype)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)  # (batch_size, hidden_dim)
        return context_vector


class VolatilityAwareLoss(tf.keras.losses.Loss):
    """
    Custom loss function that considers both price accuracy and volatility matching.
    """
    def __init__(self, volatility_weight=0.3, name='volatility_aware_loss'):
        super().__init__(name=name)
        self.volatility_weight = volatility_weight

    def call(self, y_true, y_pred):
        # Price prediction loss (MSE)
        price_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Volatility matching loss
        true_volatility = tf.math.reduce_std(y_true, axis=1)
        pred_volatility = tf.math.reduce_std(y_pred, axis=1)
        volatility_loss = tf.reduce_mean(tf.square(true_volatility - pred_volatility))
        
        # Combine losses
        total_loss = price_loss + self.volatility_weight * volatility_loss
        return total_loss


def create_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training"""
    X_seq, y_seq = [], []
    n_samples = len(X)
    
    if n_samples < sequence_length:
        raise ValueError(f"Not enough samples ({n_samples}) for sequence length {sequence_length}")
    
    for i in range(0, n_samples - sequence_length + 1, step):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length - 1])
    
    return np.array(X_seq), np.array(y_seq)


def create_model(
    input_shape: tuple,
    lstm_units: list,
    learning_rate: float,
    dropout_rate: float = 0.2
) -> Sequential:
    """
    LSTM model with attention mechanism and volatility awareness.
    """
    model = Sequential([
        Input(shape=input_shape),

        # First LSTM block with residual connection
        LSTM(lstm_units[0], return_sequences=True, 
             kernel_initializer="he_normal",
             recurrent_initializer="orthogonal",
             kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Second LSTM block with residual connection
        LSTM(lstm_units[1], return_sequences=True,
             kernel_initializer="he_normal",
             recurrent_initializer="orthogonal",
             kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Third LSTM block with attention
        LSTM(lstm_units[2], return_sequences=True,
             kernel_initializer="he_normal",
             recurrent_initializer="orthogonal",
             kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Attention layer
        AttentionLayer(),

        # Dense layers with residual connections
        Dense(256, activation="relu", 
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(128, activation="relu",
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Output layer with linear activation for returns prediction
        Dense(PREDICTION_DAYS, activation="linear", dtype='float32')
    ])

    # Use Adam with gradient clipping and a lower learning rate
    optimizer = Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    # custom volatility-aware loss
    model.compile(
        optimizer=optimizer,
        loss=VolatilityAwareLoss(volatility_weight=0.3),
        metrics=["mae"],
        jit_compile=True
    )
    return model


def load_trained_model(symbol: str) -> Optional[tf.keras.Model]:
    """Load trained model and scalers for a given symbol"""
    model_paths = [
        f"{MODEL_DIR}/{symbol}_best.keras",
        f"{MODEL_DIR}/{symbol}_best.h5"
    ]
    
    for model_path in model_paths:
        abs_path = os.path.abspath(model_path)
        logger.info(f"Attempting to load model from: {abs_path}")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {abs_path}")
            continue
            
        try:
            logger.info("Checking file permissions...")
            if not os.access(model_path, os.R_OK):
                logger.error(f"No read permission for model file at {abs_path}")
                continue
                
            logger.info("File exists and is readable. Attempting to load model...")
            
            # First try loading with custom objects
            try:
                logger.info("Attempting to load with custom objects...")
                model = load_model(
                    model_path,
                    custom_objects={"AttentionLayer": AttentionLayer},
                    compile=False  # Don't compile the model on load
                )
                logger.info("Successfully loaded model with custom objects")
            except Exception as e:
                logger.warning(f"Failed to load with custom objects: {str(e)}")
                logger.info("Attempting to load without custom objects...")
                # If that fails, try loading without custom objects
                model = load_model(model_path, compile=False)
                logger.info("Successfully loaded model without custom objects")
                
            logger.info("Recompiling model...")
            # Recompile the model with the same settings
            model.compile(
                optimizer=Adam(learning_rate=0.0001, clipnorm=0.5),
                loss=tf.keras.losses.Huber(delta=0.05),
                metrics=["mae"]
            )
            
            logger.info(f"Successfully loaded and compiled model from {abs_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {abs_path}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    logger.error(f"No valid model found for {symbol} in any format")
    return None


def predict_next_week(
    model: tf.keras.Model,
    target_scaler: MinMaxScaler,
    last_sequence_features: np.ndarray, # Should have shape (1, SEQUENCE_LENGTH, n_features)
    last_close_price: float,
    *, origin_date: str
) -> Dict:
    """
    Predict the next PREDICTION_DAYS closing prices.

    Args:
        model: The trained Keras model.
        target_scaler: The scaler used for the target variable (returns).
        last_sequence_features: The last sequence of scaled features (shape: (1, SEQUENCE_LENGTH, n_features)).
        last_close_price: The closing price of the last day in the sequence.

    Returns:
        A dictionary containing prediction dates, predicted prices, and confidence intervals.
    """
    try:
   

        # Make scaled return predictions
        # model.predict will output (1, PREDICTION_DAYS), we take the first (and only) sample
        scaled_return_predictions = model.predict(last_sequence_features, verbose=0)[0] # Output shape (PREDICTION_DAYS,)

        # Inverse transform scaled return predictions to get unscaled returns
        unscaled_return_predictions = target_scaler.inverse_transform(scaled_return_predictions.reshape(-1, 1)).flatten()

        # Calculate predicted prices from returns and the last close price
        predicted_prices=prices_from_returns([last_close_price],[unscaled_return_predictions])[0].tolist()

        # Generate prediction dates (starting from the day after the last close)
        prediction_dates=estimated_weekday_dates(origin_date,PREDICTION_DAYS)

        # Calculate a simple confidence interval (e.g., based on historical volatility or a fixed percentage)
        confidence_intervals = [abs(price * 0.02) for price in predicted_prices] #
        
        return {
            "dates": prediction_dates, # Placeholder dates
            "predictions": predicted_prices,
            "confidence_intervals": confidence_intervals,
            "date_basis": "estimated weekdays from forecast origin; exchange holidays not modeled"
        }

    except Exception as e:
        logger.error(f"Error in predict_next_week: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def predict_next_days(symbol: str, last_date: datetime) -> Tuple[List[str], List[float], List[float]]:
    """Generate predictions for the next PREDICTION_DAYS"""
    
 
