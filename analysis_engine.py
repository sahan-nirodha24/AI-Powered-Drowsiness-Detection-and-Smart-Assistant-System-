import pandas as pd
import numpy as np
from firebase_admin import db
import firebase_admin
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

# Suppress statsmodels warnings for clean console output
warnings.filterwarnings("ignore", module="statsmodels")

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    pass

class AnalyticsEngine:
    def __init__(self):
        pass

    def load_session_from_firebase(self, session_id):
        """Fetch session data from Firebase and return as pandas DataFrame"""
        if not firebase_admin._apps:
            print("Firebase not initialized.")
            return None
            
        ref = db.reference(f'/obd2_data/{session_id}')
        data = ref.get()
        
        if not data:
            return None
            
        # Convert dictionary of pushes to list of dictionaries
        records = list(data.values())
        df = pd.DataFrame(records)
        
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            df.sort_index(inplace=True)
            
        return df

    def apply_smoothing(self, df, parameter, window=5):
        """Apply Simple Moving Average (SMA) and Exponential Moving Average (EMA)"""
        if parameter not in df.columns:
            return df
            
        df[f'{parameter}_SMA'] = df[parameter].rolling(window=window).mean()
        df[f'{parameter}_EMA'] = df[parameter].ewm(span=window, adjust=False).mean()
        return df

    def detect_anomalies(self, df, parameter, threshold=3):
        """Detect anomalies using the 3-sigma rule (Z-score based on rolling stats)"""
        if parameter not in df.columns:
            return df
            
        rolling_mean = df[parameter].rolling(window=10).mean()
        rolling_std = df[parameter].rolling(window=10).std()
        
        # Calculate Z-score
        z_scores = (df[parameter] - rolling_mean) / rolling_std
        
        # Flag anomalies where Z-score is greater than threshold
        df[f'{parameter}_Anomaly'] = np.abs(z_scores) > threshold
        
        return df

    def forecast_trend(self, df, parameter, steps=10):
        """Forecast future values using ARIMA"""
        if parameter not in df.columns or len(df.dropna(subset=[parameter])) < 20:
            return None, None
            
        series = df[parameter].dropna()
        
        try:
            # Fit a simple ARIMA model (p=1, d=1, q=1)
            model = ARIMA(series, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Forecast next 'steps'
            forecast = model_fit.forecast(steps=steps)
            
            # Create future timestamps based on average frequency
            time_diffs = series.index.to_series().diff().dropna()
            if not time_diffs.empty:
                avg_diff = time_diffs.mean()
                last_time = series.index[-1]
                future_times = [last_time + (avg_diff * i) for i in range(1, steps + 1)]
                return future_times, forecast.values
        except Exception as e:
            print(f"Forecasting error: {e}")
            
        return None, None
