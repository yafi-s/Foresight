# Stock Price Prediction Model Training
# Handles model creation, training, and evaluation

import argparse
import os
import numpy as np
import joblib
import logging
import traceback
from datetime import datetime
from typing import Tuple, Optional, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, BatchNormalization, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data_loader import load_stock_data, prepare_features, PREDICTION_DAYS
from model import AttentionLayer
from causal import prepare_split, prices_from_returns
from artifact_protocol import write_bundle_metadata
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

# Performance Tuning
# Enable global XLA JIT compilation     
tf.config.optimizer.set_jit(True)

# Configure thread pools to use all available CPU cores
NUM_THREADS = min(2, cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel('ERROR')
tf.keras.mixed_precision.set_global_policy('mixed_float16')

SEQUENCE_LENGTH = 30  # Days of historical data used for each prediction

class TrainingMonitor(tf.keras.callbacks.Callback):
    """Monitor training progress and calculate metrics"""
    def __init__(self, X_val, y_val, feature_cols, target_scaler, *args, **kwargs):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.feature_cols = feature_cols
        self.target_scaler = target_scaler
        self.best_mape = float('inf')
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # Get predictions
        y_val_pred_scaled = self.model.predict(self.X_val, verbose=0)
        
        # Unscale returns
        y_val_true_unscaled_returns = self.target_scaler.inverse_transform(self.y_val.reshape(-1, 1)).reshape(self.y_val.shape)
        y_val_pred_unscaled_returns = self.target_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).reshape(y_val_pred_scaled.shape)
        
        # Calculate prices
        last_close_feature_index = self.X_val.shape[2] - 1
        last_closes_of_sequences = self.X_val[:, -1, last_close_feature_index]
        y_val_true_unscaled_prices = last_closes_of_sequences[:, np.newaxis] * (1 + y_val_true_unscaled_returns)
        y_val_pred_unscaled_prices = last_closes_of_sequences[:, np.newaxis] * (1 + y_val_pred_unscaled_returns)
        
        # Calculate metrics
        price_metrics = calculate_metrics(y_val_true_unscaled_prices, y_val_pred_unscaled_prices)
        
        # Log metrics
        logger.info(f"\nEpoch {epoch + 1} Metrics:")
        logger.info(f"Overall MAPE: {price_metrics['mape']:.2f}%")
        logger.info(f"Overall MAE: {price_metrics['mae']:.2f}")
        logger.info(f"Overall RMSE: {price_metrics['rmse']:.2f}")
        
        # Track best MAPE
        if price_metrics['mape'] < self.best_mape:
            self.best_mape = price_metrics['mape']
            self.best_epoch = epoch + 1
            logger.info(f"New best MAPE: {self.best_mape:.2f}% at epoch {self.best_epoch}")

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate prediction accuracy metrics (MAPE, MAE, RMSE) for each day"""
    mape_per_day = []
    mae_per_day = []
    rmse_per_day = []

    for d in range(y_true.shape[1]):
        t = y_true[:, d]
        p = y_pred[:, d]
        epsilon = 1e-10

        mape = np.mean(np.abs((t - p) / (np.abs(t) + epsilon))) * 100
        mae  = np.mean(np.abs(t - p))
        rmse = np.sqrt(np.mean((t - p) ** 2))

        mape_per_day.append(mape)
        mae_per_day.append(mae)
        rmse_per_day.append(rmse)

    return {
        "mape": np.mean(mape_per_day),
        "mae":  np.mean(mae_per_day),
        "rmse": np.sqrt(np.mean((y_true - y_pred)**2)), # Calculate overall RMSE directly
        "mape_per_day": mape_per_day,
        "mae_per_day": mae_per_day,
        "rmse_per_day": rmse_per_day,
    }

def create_model(input_shape: tuple, lstm_units: list, learning_rate: float, dropout_rate: float = 0.2) -> Sequential:
    """Create LSTM model with attention mechanism and dense layers"""
    model = Sequential([
        Input(shape=input_shape),

        # First LSTM block for trend recognition
        LSTM(lstm_units[0], return_sequences=True, 
             kernel_initializer="he_normal",
             recurrent_initializer="orthogonal",
             kernel_regularizer=regularizers.l2(1e-5)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Second LSTM block for market patterns
        LSTM(lstm_units[1], return_sequences=True,
             kernel_initializer="he_normal",
             recurrent_initializer="orthogonal",
             kernel_regularizer=regularizers.l2(1e-5)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Attention layer for pattern focus
        AttentionLayer(),

        # Market pattern branch
        Dense(256, activation="relu", 
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(1e-5)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Volatility branch
        Dense(128, activation="tanh",
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(1e-5)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Market noise branch
        Dense(64, activation="relu",
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(1e-5)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Output layer with linear activation
        Dense(PREDICTION_DAYS, activation="linear", dtype='float32')
    ])

    # Use Adam with gradient clipping
    optimizer = Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss="huber",
        metrics=["mae"],
        jit_compile=True
    )
    return model

def train_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                symbol: str, feature_cols: List[str], epochs: int = 100, batch_size: int = 64,
                learning_rate: float = 0.001, lstm_units: list = [256, 128], dropout_rate: float = 0.15,
                feature_scaler: StandardScaler = None, target_scaler: StandardScaler = None) -> tf.keras.Model:
    """Train the model with adaptive sample weights and save checkpoints"""
    try:
        logger.info(f"Training model for {symbol}")
        
        model = create_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            lstm_units=lstm_units,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate
        )

        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                mode="min",
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            TrainingMonitor(X_val, y_val, feature_cols, target_scaler)
        ]

        # Calculate adaptive sample weights
        sample_weights = np.ones(len(X_train))
        if 'Volatility' in feature_cols:
            vol_idx = feature_cols.index('Volatility')
            volatilities = X_train[:, -1, vol_idx]
            vol_min, vol_max = volatilities.min(), volatilities.max()
            if vol_max > vol_min:
                sample_weights = 0.5 + 1.5 * (volatilities - vol_min) / (vol_max - vol_min)
        
        # Train with adaptive sample weights
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
            sample_weight=sample_weights
        )

        # Save the best model
        try:
            logger.info("Saving the best model...")
            model.save(f"models/{symbol}_best.keras", save_format="keras")
            model.save(f"models/{symbol}_best.h5", save_format="h5")
            logger.info("Model saved successfully in both .keras and .h5 formats")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        # Save scalers
        joblib.dump(feature_scaler, f"models/{symbol}_feature_scaler.joblib")
        joblib.dump(target_scaler, f"models/{symbol}_target_scaler.joblib")

        # Enhanced training history logging
        with open(f"models/{symbol}_history.txt", "w") as f:
            f.write("Training History:\n")
            f.write(f"Final train MAE (loss): {history.history['loss'][-1]:.4f}\n")
            f.write(f"Final val   MAE (loss): {history.history['val_loss'][-1]:.4f}\n")
            f.write(f"Final train metric MAE: {history.history['mae'][-1]:.4f}\n")
            f.write(f"Final val   metric MAE: {history.history['val_mae'][-1]:.4f}\n\n")
            f.write(f"Model Architecture:\n")
            f.write(f"LSTM Units: {lstm_units}\n")
            f.write(f"Dropout Rate: {dropout_rate}\n")
            f.write(f"Learning Rate: {learning_rate}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Sequence Length: {SEQUENCE_LENGTH}\n")
            f.write(f"Train samples: {X_train.shape[0]}\n")
            f.write(f"Val   samples: {X_val.shape[0]}\n")

        logger.info(f"Model training completed for {symbol}")
        return model

    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        logger.error(traceback.format_exc())
        raise

def main():
    """Main training pipeline: load data, train model, evaluate results"""
    parser = argparse.ArgumentParser(description="Train a stock prediction model.")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol to train the model for (e.g., NVDA)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train for (default: 25)")
    args = parser.parse_args()

    logger.info(f"Starting training for {args.symbol} with {args.epochs} epochs")

    # Load and prepare data
    try:
        df = load_stock_data(symbol=args.symbol, force_refresh=True)
        if df.empty:
            logger.error(f"Could not load data for {args.symbol}.")
            return
        logger.info(f"Loaded data shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Prepare features and target
    X_df_processed, y_df_processed, feature_cols = prepare_features(df.copy())
    logger.info(f"X_df_processed shape: {X_df_processed.shape}, y_df_processed shape: {y_df_processed.shape}")

    positions = df.index.get_indexer(X_df_processed.index)
    split = prepare_split(
        X_df_processed.values, y_df_processed.values,
        df.loc[X_df_processed.index, 'Close'].values, positions,
        sequence_length=SEQUENCE_LENGTH,
    )
    X_train, X_val = split.X_train, split.X_val
    y_train, y_val = split.y_train, split.y_val
    feature_scaler, target_scaler = split.feature_scaler, split.target_scaler
    logger.info("Causal v2 split: %d train, %d validation, %d horizon-overlap samples purged",
                len(X_train), len(X_val), split.purged_samples)

    # Train model
    model = train_model(
        symbol=args.symbol,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_cols=feature_cols,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        epochs=args.epochs
    )
    # Marker is written only after a causal split and complete model/scaler save.
    write_bundle_metadata('models',args.symbol,feature_cols)

    # Evaluate on validation set
    y_val_pred_scaled = model.predict(X_val, verbose=0)

    # Unscale returns
    y_val_true_unscaled_returns = target_scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_val_pred_unscaled_returns = target_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).reshape(y_val_pred_scaled.shape)

    # Calculate prices
    last_close_feature_index = X_train.shape[2] - 1
    last_closes_of_sequences = X_val[:, -1, last_close_feature_index]
    y_val_true_unscaled_prices = last_closes_of_sequences[:, np.newaxis] * (1 + y_val_true_unscaled_returns)
    y_val_pred_unscaled_prices = last_closes_of_sequences[:, np.newaxis] * (1 + y_val_pred_unscaled_returns)

    # Calculate and log metrics
    logger.info("\nCalculating final validation metrics on unscaled prices:")
    price_metrics = calculate_metrics(y_val_true_unscaled_prices, y_val_pred_unscaled_prices)

    logger.info(f"Overall MAPE: {price_metrics['mape']:.2f}%")
    logger.info(f"Overall MAE: {price_metrics['mae']:.2f}")
    logger.info(f"Overall RMSE: {price_metrics['rmse']:.2f}")

    logger.info("\nMetrics per prediction day (unscaled prices):")
    for d in range(PREDICTION_DAYS):
        logger.info(
            f"  Day {d+1}: MAPE={price_metrics['mape_per_day'][d]:.2f}%, "
            f"MAE={price_metrics['mae_per_day'][d]:.2f}, "
            f"RMSE={price_metrics['rmse_per_day'][d]:.2f}"
        )

    # Save metrics
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"{metrics_dir}/{args.symbol}_metrics_{timestamp}.txt"

    with open(metrics_filename, "w") as f:
        f.write(f"Symbol: {args.symbol}\n")
        last_train_close = float(X_train[-1, -1, -1])
        f.write("Evaluation protocol: causal-v2; validation used for early stopping, not untouched test\n")
        f.write(f"Purged overlapping samples: {split.purged_samples}\n")
        f.write(f"Last Close (train end): ${last_train_close:.2f}\n")
        f.write(f"Overall MAPE: {price_metrics['mape']:.2f}%\n")
        f.write(f"Overall MAE: {price_metrics['mae']:.2f}\n")
        f.write(f"Overall RMSE: {price_metrics['rmse']:.2f}\n")
        f.write("\nMetrics per prediction day (unscaled prices):\n")
        for d in range(PREDICTION_DAYS):
            f.write(
                f"  Day {d+1}: MAPE={price_metrics['mape_per_day'][d]:.2f}%, "
                f"MAE={price_metrics['mae_per_day'][d]:.2f}, "
                f"RMSE={price_metrics['rmse_per_day'][d]:.2f}\n"
            )

    logger.info(f"Training complete. Metrics saved to {metrics_filename}")

if __name__ == "__main__":
    main()
