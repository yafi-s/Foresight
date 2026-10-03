"""Chronological training preparation. No model, network, or runtime side effects."""
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    train_origins: np.ndarray
    val_origins: np.ndarray
    purged_samples: int


def prepare_split(features, returns, closes, positions, sequence_length=30, train_fraction=0.8):
    """Fit transforms only on training observations; purge overlapping labels.

    A row's close is observed at that row, never sequence_length-1 rows later.
    Targets are cumulative returns from the final input timestamp. Positions are
    original row numbers, so gaps in the feature table cannot hide label overlap.
    Validation is for model selection; a separate untouched test is still needed.
    """
    features = np.asarray(features, dtype=float)
    returns = np.asarray(returns, dtype=float)
    closes = np.asarray(closes, dtype=float)
    positions = np.asarray(positions, dtype=int)
    n = len(features)
    if (features.ndim != 2 or returns.ndim != 2 or closes.shape != (n,)
            or positions.shape != (n,) or len(returns) != n
            or sequence_length < 1 or not 0 < train_fraction < 1
            or n < sequence_length + 2 or np.any(np.diff(positions) <= 0)
            or not all(np.isfinite(x).all() for x in (features, closes))):
        raise ValueError("Invalid aligned chronological arrays")
    origins = np.arange(sequence_length - 1, n)
    origins = origins[np.isfinite(returns[origins]).all(axis=1)]
    split = int(len(origins) * train_fraction)
    if split == 0 or split == len(origins):
        raise ValueError("Insufficient samples for both partitions")
    val_indices = origins[split:]
    first_val_position = positions[val_indices[0]]
    # All training target timestamps must precede the first validation origin.
    train_indices = origins[:split]
    train_indices = train_indices[positions[train_indices] + returns.shape[1] < first_val_position]
    if not len(train_indices):
        raise ValueError("No training samples remain after horizon purge")
    feature_scaler = StandardScaler().fit(features[:train_indices[-1] + 1])
    target_scaler = StandardScaler().fit(returns[train_indices].reshape(-1, 1))
    scaled = feature_scaler.transform(features)
    # Raw current close is retained for price reconstruction and input parity.
    augmented = np.column_stack((scaled, closes))
    def windows(indices):
        return np.stack([augmented[i - sequence_length + 1:i + 1] for i in indices])
    def targets(indices):
        raw = returns[indices]
        return target_scaler.transform(raw.reshape(-1, 1)).reshape(raw.shape)
    return Split(windows(train_indices), targets(train_indices), windows(val_indices),
                 targets(val_indices), feature_scaler, target_scaler,
                 positions[train_indices], positions[val_indices], split - len(train_indices))


def prices_from_returns(closes, cumulative_returns):
    """Every horizon is relative to the same observed origin close."""
    return np.asarray(closes)[:, None] * (1 + np.asarray(cumulative_returns))


def prepare_inference(features, closes, feature_scaler, sequence_length=30):
    """Latest observed window with the same per-row close channel as training."""
    features=np.asarray(features,dtype=float)
    closes=np.asarray(closes,dtype=float)
    if (features.ndim!=2 or closes.shape!=(len(features),) or sequence_length<1
            or len(features)<sequence_length or not np.isfinite(features).all()
            or not np.isfinite(closes).all()):
        raise ValueError('Invalid inference observations')
    scaled=feature_scaler.transform(features[-sequence_length:])
    return np.column_stack((scaled,closes[-sequence_length:]))[None,:,:]


def estimated_weekday_dates(origin_date, horizons):
    """Origin-based date estimates, explicitly excluding an exchange calendar."""
    current=datetime.fromisoformat(str(origin_date)[:10]).date()
    dates=[]
    while len(dates)<horizons:
        current+=timedelta(days=1)
        if current.weekday()<5:
            dates.append(current.isoformat())
    return dates
