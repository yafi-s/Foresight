# Stock Prediction API
# FastAPI server for model predictions and data management

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import uvicorn
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sqlmodel import Session
from sqlalchemy import select
import os
import joblib
import logging
import traceback
import pandas as pd
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data_loader import load_stock_data, prepare_features
from causal import prepare_inference
from artifact_protocol import validate_bundle
from model import (
    load_trained_model,
    predict_next_week,
    SEQUENCE_LENGTH,
    PREDICTION_DAYS
)
from models import StockData, Prediction
from database import get_session, create_db_and_tables

# Get the absolute path to the models directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

app = FastAPI(
    title="Stock Prediction API",
    description="API for predicting next-day stock prices using LSTM/TCN models",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Stock Price Prediction API"}

@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy"}

@app.get("/api/load/{symbol}")
def load_data(symbol: str, refresh: bool = False) -> Dict:
    """
    Load or refresh stock data for a symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        refresh: Force refresh data from yfinance
        
    Returns:
        Dict with status and data info
    """
    try:
        df = load_stock_data(symbol, force_refresh=refresh)
        return {
            "status": "success",
            "symbol": symbol,
            "rows": len(df),
            "last_updated": df.index[-1].strftime("%Y-%m-%d"),
            "latest_close": float(df["Close"].iloc[-1])
        }
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def check_model_exists(symbol: str) -> bool:
    """
    Check if a trained model exists for the given symbol.

    """
    model_path = os.path.join(MODEL_DIR, f"{symbol}_best.keras")
    return os.path.exists(model_path)

def get_training_command(symbol: str) -> str:
    """
    Get the command to train a model for the given symbol.
    """
    return f"python train.py --symbol {symbol}"

@app.get("/api/predict/{symbol}")
def predict(symbol: str, session: Session = Depends(get_session)) -> Dict:
    """
    Predict next week's closing prices for a symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        
    Returns:
        Dict with predictions and metadata
    """
    try:
        logger.info(f"Starting prediction for {symbol}")
        
        # Check if model exists
        if not check_model_exists(symbol):
            return {
                "symbol": symbol,
                "lastActualClose": 0,
                "lastUpdated": "",
                "mape": None,
                "predictions": {
                    "dates": [],
                    "predictions": [],
                    "confidenceIntervals": []
                },
                "error": True,
                "message": f"Model not available for {symbol}. Please refer to the README for training instructions."
            }
        
        # Old/mixed artifacts do not have the repaired training input contract.
        validate_bundle(MODEL_DIR,symbol)

        # Load data
        logger.info("Loading stock data...")
        df = load_stock_data(symbol)
        logger.info(f"DataFrame shape: {df.shape}")
        
        # Prepare features
        logger.info("Preparing features...")
        features, _, feature_cols = prepare_features(df.copy(),include_targets=False)
        if features.empty or features.index[-1]!=df.index[-1]:
            raise HTTPException(status_code=400,detail='Latest observation has incomplete features')
        if len(features)<SEQUENCE_LENGTH:
            raise HTTPException(status_code=400,detail='Not enough data for prediction')
        logger.info(f"Feature columns: {feature_cols}")
        
        # Load feature scaler
        feature_scaler_path = os.path.join(MODEL_DIR, f"{symbol}_feature_scaler.joblib")
        if not os.path.exists(feature_scaler_path):
            raise HTTPException(status_code=404, detail="Feature scaler not found")
        feature_scaler = joblib.load(feature_scaler_path)
        
        closes=df.loc[features.index,'Close'].to_numpy()
        last_sequence=prepare_inference(features.to_numpy(),closes,feature_scaler,SEQUENCE_LENGTH)
        last_close=float(closes[-1])
        origin_date=features.index[-1].strftime('%Y-%m-%d')
        
        # Load model and target scaler
        model = load_trained_model(symbol)
        target_scaler_path = os.path.join(MODEL_DIR, f"{symbol}_target_scaler.joblib")
        target_scaler = joblib.load(target_scaler_path)
        
        # Get predictions
        predictions = predict_next_week(model,target_scaler,last_sequence,last_close,origin_date=origin_date)
        
        if not predictions or not predictions.get("predictions"):
            raise HTTPException(status_code=500, detail="Failed to generate predictions")
        
        # Store predictions in database
        for date, pred_price in zip(predictions["dates"], predictions["predictions"]):
            prediction = Prediction(
                symbol=symbol,
                date=datetime.strptime(date, "%Y-%m-%d"),
                predicted_close=pred_price,
                actual_close=0.0  # Will be updated when actual price is known
            )
            session.add(prediction)
        session.commit()
        
        return {
            "symbol": symbol,
            "lastActualClose": last_close,
            "lastUpdated": df.index[-1].strftime("%Y-%m-%d"),
            "forecastOrigin": origin_date,
            "dateBasis": predictions['date_basis'],
            "mape": None,  # Will be calculated when actual prices are known
            "predictions": {
                "dates": predictions["dates"],
                "predictions": predictions["predictions"],
                "confidenceIntervals": predictions["confidence_intervals"]
            },
            "error": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{symbol}")
def get_prediction_history(symbol: str, session: Session = Depends(get_session)) -> Dict:
    """
    Get historical predictions and their accuracy for a symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        
    Returns:
        Dict with historical predictions and accuracy metrics
    """
    try:
        # Get all predictions for the symbol
        statement = (
            select(Prediction)
            .where(Prediction.symbol == symbol)
            .order_by(Prediction.date.desc())
        )
        predictions = session.exec(statement).all()
        
        if not predictions:
            return {
                "symbol": symbol,
                "history": [],
                "error": False,
                "message": "No historical predictions found"
            }
        
        # Calculate accuracy metrics
        history = []
        for pred in predictions:
            if pred.actual_close > 0:  # Only include predictions where we have actual prices
                mape = abs((pred.actual_close - pred.predicted_close) / pred.actual_close) * 100
                history.append({
                    "date": pred.date.strftime("%Y-%m-%d"),
                    "predicted": pred.predicted_close,
                    "actual": pred.actual_close,
                    "mape": mape
                })
        
        return {
            "symbol": symbol,
            "history": history,
            "error": False
        }
        
    except Exception as e:
        logger.error(f"Error in get_prediction_history: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update-actual-prices")
def update_actual_prices(session: Session = Depends(get_session)) -> Dict:
    """
    Update actual prices for past predictions.
    This should be run daily to update the accuracy of past predictions.
    """
    try:
        # Get all predictions without actual prices
        statement = (
            select(Prediction)
            .where(Prediction.actual_close == 0)
            .order_by(Prediction.date)
        )
        predictions = session.exec(statement).all()
        
        updated = 0
        for pred in predictions:
            # Load stock data for the prediction date
            df = load_stock_data(pred.symbol)
            if pred.date.strftime("%Y-%m-%d") in df.index:
                pred.actual_close = float(df.loc[pred.date.strftime("%Y-%m-%d"), "Close"])
                updated += 1
        
        session.commit()
        
        return {
            "status": "success",
            "updated": updated,
            "message": f"Updated {updated} predictions with actual prices"
        }
        
    except Exception as e:
        logger.error(f"Error in update_actual_prices: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Use reload=False when deploying or running in production
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
