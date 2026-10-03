**Evaluation status:** The causal-v2 repair removes full-dataset scaler fitting,
future-close inputs, horizon shifts across missing feature rows, and incorrect
cumulative-return compounding. Historical checkpoints and low-MAPE results are
legacy artifacts and do not demonstrate held-out accuracy. Ten offline tests
pass; no model was retrained. Serving rejects legacy bundles until causal-v2
retraining creates a matching protocol marker. See [the protocol and evidence gaps](docs/CAUSAL_EVALUATION.md).

    # Stock Price Prediction 

    ##  IMPORTANT DISCLAIMER 

    **This application is for educational and research purposes only. It is NOT intended for actual stock trading or investment decisions.**

    - The predictions provided by this application are based on historical data and machine learning models, which are inherently imperfect and cannot account for all market variables.
    - Past performance is not indicative of future results.
    - The program and developer:
    - Are not responsible for any financial losses incurred from using this application
    - Do not guarantee the accuracy of predictions
    - Do not endorse any trading decisions based on these predictions

    **USE AT YOUR OWN RISK. Always consult with qualified financial advisors before making any investment decisions.**

    ## Overview

    A full-stack web application that demonstrates machine learning concepts by predicting next-day closing prices for U.S. stocks using LSTM neural networks. This project is intended for educational purposes to understand:
    - Machine learning model development
    - Time series analysis
    - Full-stack web development
    - API integration
    - Data visualization

    ## Features

    - Real-time stock data retrieval using yfinance
    - Advanced LSTM-based price prediction model with attention mechanism
    - Interactive price charts with predictions
    - SQLite database for efficient data caching and management
    - React frontend with Material-UI
    - FastAPI backend with automatic API documentation
    - Historical prediction tracking and accuracy analysis

    ## Tech Stack

    ### Backend
    - Python 3.12
    - FastAPI
    - TensorFlow/Keras
    - yfinance
    - SQLModel
    - scikit-learn

    ### Frontend
    - React (Vite)
    - TypeScript
    - Material-UI
    - Recharts
    - Axios

    ## LSTM Model Architecture

    The model uses a sophisticated LSTM architecture with attention mechanism:

    1. **Input Layer**
       - 30-day sequence of 8 technical features
       - Features include: Returns, Volatility, High-Low Range, Volume Change, Price Momentum, Volume Momentum, BB Position, MA5

    2. **LSTM Layers**
       - First LSTM block (256 units): Trend recognition
       - Second LSTM block (128 units): Market pattern analysis
       - Third LSTM block with attention: Focus on important time steps
       - All LSTM layers use return_sequences=True for temporal pattern preservation

    3. **Attention Mechanism**
       - Custom AttentionLayer for focusing on significant time steps
       - Helps model identify critical patterns in the sequence

    4. **Dense Layers**
       - 256-unit dense layer with ReLU activation
       - 128-unit dense layer with ReLU activation
       - Output layer: 5-day price predictions

    5. **Regularization**
       - Batch Normalization after each major layer
       - Dropout (0.2) for preventing overfitting
       - L2 regularization (1e-4) on weights

    6. **Training Features**
       - Custom VolatilityAwareLoss function
       - Adam optimizer with gradient clipping
       - Mixed precision training
       - JIT compilation for performance

    ## Data Management

    The application uses SQLite database for efficient data management:

    1. **StockData Table**
       - Stores historical OHLCV data
       - Caches data to reduce API calls
       - Indexed by symbol and date

    2. **Prediction Table**
       - Stores model predictions
       - Tracks prediction accuracy
       - Enables historical analysis

    ## Setup

    ### Backend

    1. Install dependencies:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

    2. Start the FastAPI server:
    ```bash
    cd backend
    python main.py
    ```

    ### Frontend (create new terminal)

    1. Install dependencies:
    ```bash
    cd frontend
    npm install
    ```

    2. Start the development server:
    ```bash
    npm run dev
    ```

    The frontend will be available at http://localhost:5173

    ## Usage

    1. Train a model for a specific stock:
    ```bash
    python train.py --symbol SYMBOL
    Pretrained Models: AAPL, NVDA, NFLX, SOFI, UL, INAB, TM, MSFT, TSLA, and GOOGL
    ```


    2. Access the web interface at http://localhost:5173
    3. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT)
    4. Click "Predict" to get 5-day price predictions

    ## Model Management

    ```bash
    # Train a new model (create a new terminal)
    cd backend
    python train.py --symbol SYMBOL

    # Clear existing model files
    cd backend
    Remove-Item models\SYMBOL_*

    # Clear existing metric files
    cd backend
    Remove-Item metrics\SYMBOL_*
    ```

    ## License

    MIT License

    def calculate_mape(actual, predicted):
        return np.mean(np.abs((actual - predicted) / actual)) * 100     
