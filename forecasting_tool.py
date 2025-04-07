import pandas as pd
import streamlit as st
from prophet import Prophet
from pmdarima import auto_arima
from xgboost import XGBRegressor
import xgboost as xgb
from flaml import AutoML
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import plotly.express as px
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import optuna
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf
import time
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import catboost
from tqdm import tqdm
import concurrent.futures
from scipy.stats import pearsonr
import calendar
from sklearn.metrics import mean_absolute_error, r2_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.signal import periodogram
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
from statsmodels.tsa.stattools import pacf
from sklearn.model_selection import TimeSeriesSplit
from optuna.samplers import TPESampler  # Add this import
from optuna.pruners import MedianPruner
from optuna import trial

# Enable Wide Mode (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(layout="wide", page_title="Time Series Forecasting", page_icon="📈")

# Add dark mode toggle
if "theme" not in st.session_state:
    st.session_state.theme = "light"

theme = st.radio("🌙 Theme Mode:", ["Light", "Dark"], index=0 if st.session_state.theme=="light" else 1)
st.session_state.theme = theme

if st.session_state.theme=="dark":
    st.markdown(
        """
        <style>
            body { background-color: #1E1E1E; color: white; }
            .stButton button { background-color: #56BBAF !important; }
            .stDataFrame { background-color: #2E2E2E; color: white; }
            .stTextInput input { background-color: #2E2E2E; color: white; }
            .stSelectbox select { background-color: #2E2E2E; color: white; }
            .stRadio div { color: white; }
            .stMarkdown { color: white; }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown(
        """
        <style>
            body { background-color: white; color: black; }
            .stButton button { background-color: #56BBAF !important; }
            .stDataFrame { background-color: white; color: black; }
            .stTextInput input { background-color: white; color: black; }
            .stSelectbox select { background-color: white; color: black; }
            .stRadio div { color: black; }
            .stMarkdown { color: black; }
        </style>
        """, unsafe_allow_html=True)

# ====================== ENHANCED CORE FUNCTIONS ======================
def check_stationarity(series, categories=None):
    """Enhanced stationarity check with hierarchical analysis"""
    results = {}
    
    # Global check
    adf = adfuller(series, autolag="AIC")
    kpss_result = kpss(series, regression="c", nlags="auto")
    results['global'] = {
        'adf_p': adf[1],
        'kpss_p': kpss_result[1],
        'stationarity': None
    }
    
    # Category-level checks
    if categories is not None:
        for name, group in series.groupby(categories):
            adf = adfuller(group, autolag="AIC")
            kpss_result = kpss(group, regression="c", nlags="auto")
            results[name] = {
                'adf_p': adf[1],
                'kpss_p': kpss_result[1],
                'stationarity': None
            }
    
    # Determine stationarity
    for key in results:
        adf_p = results[key]['adf_p']
        kpss_p = results[key]['kpss_p']
        if adf_p < 0.05 and kpss_p > 0.05:
            results[key]['stationarity'] = "Stationary"
        elif adf_p >= 0.05 and kpss_p <= 0.05:
            results[key]['stationarity'] = "Non-Stationary"
        else:
            results[key]['stationarity'] = "Inconclusive"
    
    return results

def preprocess_data(data, date_column, sales_column, category_columns=None):
    """Enhanced preprocessing with dynamic differencing and category handling"""
    try:
        # 1. Initial processing
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data.dropna(subset=[date_column, sales_column], inplace=True)
        data = data.rename(columns={date_column: "ds", sales_column: "y"})
        
        # 2. Temporal aggregation
        data["year_month"] = data["ds"].dt.to_period("M")
        grouping_cols = ["year_month"] + (category_columns if category_columns else [])
        data = data.groupby(grouping_cols, as_index=False)["y"].sum()
        
        # 3. Date conversion and deduplication
        data["ds"] = data["year_month"].dt.to_timestamp(how="start")
        data.drop(columns=["year_month"], inplace=True)
        subset_cols = ["ds", "y"] + (category_columns if category_columns else [])
        data = data.drop_duplicates(subset=subset_cols, keep="last")

        # 4. Prepare original data
        y_original = data.groupby("ds", as_index=False)["y"].sum().rename(columns={"y": "y_original"})
        
        # 5. Stationarity analysis
        stationarity = check_stationarity(data["y"], data[category_columns] if category_columns else None)
        
        # 6. Dynamic differencing
        diff_order = 0
        if any(v['stationarity'] == "Non-Stationary" for v in stationarity.values()):
            data_diff, diff_order = dynamic_differencing(data, "y", category_columns)
            last_value = data["y"].iloc[-diff_order:] if diff_order > 0 else None
            data = data_diff.dropna()
        else:
            last_value = None

        # 7. Visualization
        plot_stationarity_analysis(data, y_original, stationarity, diff_order)
        
        return data, last_value, y_original, diff_order > 0

    except Exception as e:
        st.error(f"Preprocessing Error: {str(e)}")
        st.error(f"Debug Info: {data.columns if 'data' in locals() else 'No data'}")
        return None, None, None, False

# ====================== ENHANCED UTILITIES ======================
def dynamic_differencing(data, column, categories=None, max_order=2):
    """Smart differencing with automatic order selection"""
    diff_data = data.copy()
    diff_order = 0
    
    while diff_order < max_order:
        stationarity = check_stationarity(diff_data[column], categories)
        if all([v['stationarity'] == "Stationary" for v in stationarity.values()]):
            break
        diff_data[column] = diff_data[column].diff().dropna()
        diff_order += 1
    
    return diff_data, diff_order

def inverse_difference(forecast_data, first_value, diff_order=1):
    """Robust inverse differencing with order handling"""
    if not isinstance(first_value, (int, float, np.ndarray)):
        raise ValueError("first_value must be numeric or array-like")
        
    if isinstance(forecast_data, pd.Series):
        forecast_data = forecast_data.to_frame(name="yhat")
    
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        if col in forecast_data.columns:
            if diff_order == 1:
                forecast_data[col] = first_value + forecast_data[col].cumsum()
            else:
                # Handle higher order differencing
                temp = np.r_[first_value, forecast_data[col]].cumsum()
                for _ in range(diff_order-1):
                    temp = temp.cumsum()
                forecast_data[col] = temp[-len(forecast_data):]
    
    return forecast_data

# ====================== ENHANCED MODEL COMPONENTS ======================
def detect_and_add_seasonalities(model, data):
    """FFT-based seasonality detection with fallback"""
    try:
        from scipy.signal import periodogram
        f, Pxx = periodogram(data["y"].dropna())
        significant_periods = f[Pxx > np.quantile(Pxx, 0.95)] * 365
        
        for period in significant_periods:
            if 7 <= period <= 365:
                name = f'custom_{int(period)}'
                fourier_order = max(3, int(period/30))
                model.add_seasonality(name=name, period=period, fourier_order=fourier_order)
    except Exception as e:
        st.warning(f"Seasonality detection failed: {str(e)}")
    
    return model

def tune_prophet(train_data, n_trials=50, timeout=3600):
    """Optuna-based hyperparameter optimization for Prophet with enhanced features"""
    def objective(trial):
        # Suggest parameters with intelligent ranges
        params = {
            'changepoint_prior_scale': trial.suggest_float(
                'changepoint_prior_scale', 1e-3, 0.5, log=True),
            'seasonality_prior_scale': trial.suggest_float(
                'seasonality_prior_scale', 0.1, 50, log=True),
            'seasonality_mode': trial.suggest_categorical(
                'seasonality_mode', ['additive', 'multiplicative']),
            'growth': trial.suggest_categorical(
                'growth', ['logistic', 'linear'])  # Optional: Tune growth mode
        }

        # Add logistic growth requirements if needed
        if params['growth'] == 'logistic':
            if 'cap' not in train_data.columns:
                trial.set_user_attr('error', 'Missing cap for logistic growth')
                return float('inf')
            if 'floor' not in train_data.columns:
                trial.set_user_attr('error', 'Missing floor for logistic growth')
                return float('inf')

        try:
            model = Prophet(
                **params,
                yearly_seasonality='auto',
                weekly_seasonality='auto',
                daily_seasonality=False,
                uncertainty_samples=0  # Disable for faster CV
            )
            
            # Add custom seasonalities from EnhancedProphet
            model = EnhancedProphet(**params).detect_seasonalities(train_data)
            
            model.fit(train_data)

            # Smart cross-validation configuration
            df_cv = cross_validation(
                model,
                initial='730 days',
                period='180 days',
                horizon='90 days',
                parallel="processes"
            )

            # Time-aware weighted RMSE
            metrics = performance_metrics(df_cv)
            weights = 1 / (metrics['horizon'] / pd.Timedelta(days=1) + 1e-6)
            weighted_rmse = np.average(metrics['rmse'], weights=weights)
            
            return weighted_rmse

        except Exception as e:
            trial.set_user_attr('error', str(e))
            return float('inf')

    # Configure study with enhanced settings
    sampler = TPESampler(seed=42, n_startup_trials=10)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            reduction_factor=3
        )
    )

    # Optimize with intelligent resource allocation
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=-1,
        gc_after_trial=True,
        show_progress_bar=True
    )

    # Generate optimization report
    best_trial = study.best_trial
    return {
    'params': best_trial.params,  # Remove custom_seasonalities from here
    'custom_seasonalities': best_trial.user_attrs.get('custom_seasonalities', []),
    'metrics': {
        'best_rmse': best_trial.value,
        'completed_trials': len(study.trials),
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    },
    'study': study
}

class EnhancedProphet(Prophet):
    def __init__(self, **kwargs):
        """Enhanced Prophet subclass with custom seasonality management"""
        # Remove custom parameters before parent initialization
        self.custom_seasonalities = kwargs.pop('custom_seasonalities', [])
        self.detected_seasonalities = []
        
        # Filter valid Prophet parameters
        valid_params = Prophet().__dict__.keys()
        prophet_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        super().__init__(**prophet_kwargs)
        
        # Add predefined custom seasonalities
        for season in self.custom_seasonalities:
            self._safe_add_seasonality(season)

    def _safe_add_seasonality(self, season_config):
        """Add seasonality with error handling"""
        try:
            self.add_seasonality(
                name=season_config['name'],
                period=season_config['period'],
                fourier_order=season_config['fourier_order']
            )
            if season_config['name'] not in self.custom_seasonalities:
                self.custom_seasonalities.append(season_config['name'])
        except Exception as e:
            print(f"Failed to add {season_config['name']}: {str(e)}")

    def detect_seasonalities(self, data, threshold_percentile=95, min_period=7, max_period=365):
        """FFT-based seasonality detection with enhanced safeguards"""
        try:
            y = data['y'].replace(0, np.nan).dropna()
            if len(y) < 2 * max_period:
                print("Insufficient data for reliable seasonality detection")
                return self

            # Calculate sampling frequency
            freq = pd.infer_freq(data['ds'])
            fs = self._get_sampling_rate(freq)
            
            # Compute periodogram
            f, Pxx = periodogram(y - y.mean(), fs=fs)
            significant = Pxx > np.percentile(Pxx, threshold_percentile)
            
            # Add validated seasonalities
            for freq_hz, power in zip(f[significant], Pxx[significant]):
                if freq_hz <= 0:
                    continue
                    
                period_days = 1 / freq_hz
                if min_period <= period_days <= max_period:
                    self._add_detected_seasonality(period_days, power)
                    
        except Exception as e:
            print(f"Seasonality detection failed: {str(e)}")
            
        return self

    def _get_sampling_rate(self, freq):
        """Convert pandas frequency to sampling rate"""
        freq_map = {
            'D': 1.0,      # Daily
            'MS': 12/365.25,  # Monthly start
            'M': 12/365.25,   # Monthly end
            'H': 24,       # Hourly
            'Q': 4/365.25   # Quarterly
        }
        return freq_map.get(freq, 365.25)  # Default to yearly

    def _add_detected_seasonality(self, period_days, power):
        """Add validated seasonality to model"""
        name = f'custom_{int(period_days)}d'
        fourier_order = self._calculate_fourier_order(period_days)
        
        if name not in self.custom_seasonalities + self.detected_seasonalities:
            try:
                self.add_seasonality(
                    name=name,
                    period=period_days,
                    fourier_order=fourier_order
                )
                self.detected_seasonalities.append({
                    'name': name,
                    'period': period_days,
                    'fourier_order': fourier_order,
                    'power': power
                })
            except ValueError as e:
                print(f"Skipping {name}: {str(e)}")

    def _calculate_fourier_order(self, period_days):
        """Dynamic Fourier order calculation"""
        return min(11, max(3, int(period_days / 7)))

    def get_seasonality_report(self):
        """Return formatted seasonality information"""
        report = {
            'builtin': [],
            'custom': [],
            'detected': []
        }
        
        # Built-in seasonalities
        for seas in self.seasonalities:
            if seas not in self.custom_seasonalities + [s['name'] for s in self.detected_seasonalities]:
                report['builtin'].append(seas)
        
        # Custom seasonalities
        report['custom'] = self.custom_seasonalities
        
        # Detected seasonalities
        report['detected'] = [s['name'] for s in self.detected_seasonalities]
        
        return report

def auto_arima_enhanced(data, seasonal_period=12, enforce_stationarity=True, explain=True):
    """
    Enhanced Auto ARIMA with:
    - Automatic seasonality detection
    - Model explainability
    - Enhanced error handling
    - Stationarity enforcement
    """
    result = {
        'model': None,
        'explanation': {},
        'warnings': []
    }
    
    try:
        # 1. Seasonality detection
        decomposition = None
        seasonal_strength = 0
        if len(data) >= 2*seasonal_period:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                decomposition = seasonal_decompose(data, period=seasonal_period, model='additive')
                seasonal_strength = 1 - np.nanvar(decomposition.resid)/np.nanvar(data)
                
        # 2. Determine seasonality
        is_seasonal = seasonal_strength > 0.6 and len(data) >= 3*seasonal_period
        
        # 3. Run auto_arima with enhanced configuration
        model = auto_arima(
            data,
            seasonal=is_seasonal,
            m=seasonal_period if is_seasonal else 1,
            d=None if enforce_stationarity else 1,
            D=None if enforce_stationarity else 1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
            max_order=10,
            information_criterion='aic',
            n_jobs=-1
        )
        
        # 4. Build explanation
        if explain:
            result['explanation'] = {
                'order': model.order,
                'seasonal_order': model.seasonal_order,
                'aic': model.aic(),
                'seasonal_strength': seasonal_strength,
                'decomposition_success': decomposition is not None,
                'used_seasonality': is_seasonal
            }
        
        result['model'] = model
        
    except Exception as e:
        result['warnings'].append(f"ARIMA failed: {str(e)}")
        # Fallback to simple ARIMA
        model = auto_arima(
            data,
            seasonal=False,
            suppress_warnings=True,
            error_action="ignore"
        )
        result['model'] = model
    
    return result

class TemporalXGBoost:
    def __init__(self, horizon=12, max_lags=24, fourier_terms=3):
        self.model = None
        self.horizon = horizon
        self.max_lags = max_lags
        self.fourier_terms = fourier_terms
        self.feature_columns = []
        self.last_state = {}
        self.freq = None  # Track data frequency
        
    def create_features(self, df):
        """Enhanced feature engineering with automatic frequency detection"""
        df = df.copy()
        
        # 1. Determine data frequency
        self.freq = pd.infer_freq(df['ds']) or 'MS'
        
        # 2. Lag features using PACF
        pacf_vals = pacf(df['y'], nlags=self.max_lags, method='ywm')
        significant_lags = np.where(np.abs(pacf_vals) > 1.96/np.sqrt(len(df)))[0]
        for lag in significant_lags:
            if lag > 0:
                df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # 3. Rolling statistics
        for window in [3, 6, 12]:
            df[f'rolling_mean_{window}'] = df['y'].rolling(window).mean()
            df[f'rolling_std_{window}'] = df['y'].rolling(window).std()
        
        # 4. Date features based on detected frequency
        dt = df['ds'].dt
        if self.freq == 'D':
            df['dayofweek'] = dt.dayofweek
            df['dayofyear'] = dt.dayofyear
        elif self.freq in ['MS', 'M']:
            df['month'] = dt.month
            df['quarter'] = dt.quarter
        
        # 5. Fourier terms
        if self.freq in ['MS', 'M']:
            for k in range(1, self.fourier_terms+1):
                df[f'sin_{k}'] = np.sin(2 * np.pi * k * dt.month/12)
                df[f'cos_{k}'] = np.cos(2 * np.pi * k * dt.month/12)
        
        # 6. Trend features
        df['trend'] = np.arange(len(df))
        
        self.feature_columns = [col for col in df.columns if col not in ['ds', 'y']]
        return df.dropna()

    def fit(self, train_data):
        """Enhanced training with automatic feature creation"""
        try:
            df = self.create_features(train_data)
            X = df[self.feature_columns]
            y = df['y']
            
            # Store last state for forecasting
            self.last_state = {
                'lags': X.filter(regex='lag_').iloc[-1].to_dict(),
                'rolling_stats': X.filter(regex='rolling_').iloc[-1].to_dict(),
                'trend': X['trend'].iloc[-1]
            }
            
            # Train model with cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            self.model = xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                early_stopping_rounds=50,
                n_jobs=-1
            )
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
            return self
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

    def predict(self, start_date, periods=None):
        """Robust recursive forecasting"""
        if periods is None:
            periods = self.horizon
            
        forecasts = []
        current_state = self.last_state.copy()
        
        for _ in range(periods):
            # Generate features for next step
            features = self._generate_next_features(current_state, start_date)
            
            # Predict and update state
            pred = self.model.predict(pd.DataFrame([features]))[0]
            forecasts.append(pred)
            self._update_state(current_state, pred, start_date)
            
            # Increment date
            start_date += pd.DateOffset(months=1)
            
        return pd.DataFrame({
            'ds': pd.date_range(
                start=self.last_state['date'] + pd.DateOffset(months=1),
                periods=periods,
                freq=self.freq
            ),
            'yhat': forecasts
        })

    def _generate_next_features(self, state, current_date):
        """Create feature vector for prediction"""
        features = {}
        features.update(state['lags'])
        features.update(state['rolling_stats'])
        
        # Date-based features
        dt = current_date
        if self.freq == 'D':
            features.update({
                'dayofweek': dt.dayofweek,
                'dayofyear': dt.dayofyear
            })
        elif self.freq in ['MS', 'M']:
            features.update({
                'month': dt.month,
                'quarter': (dt.month-1)//3 + 1
            })
        
        # Fourier terms
        if self.freq in ['MS', 'M']:
            for k in range(1, self.fourier_terms+1):
                features[f'sin_{k}'] = np.sin(2 * np.pi * k * dt.month/12)
                features[f'cos_{k}'] = np.cos(2 * np.pi * k * dt.month/12)
        
        # Trend features
        features['trend'] = state['trend'] + 1
        
        return features

    def _update_state(self, state, new_value, current_date):
        """Update state for next prediction"""
        # Update lags
        for lag in sorted([int(k.split('_')[1]) for k in state['lags'].keys()], reverse=True):
            if lag == 1:
                state['lags'][f'lag_1'] = new_value
            else:
                state['lags'][f'lag_{lag}'] = state['lags'].get(f'lag_{lag-1}', new_value)
        
        # Update rolling stats
        for window in [3, 6, 12]:
            state['rolling_stats'][f'rolling_mean_{window}'] = (
                state['rolling_stats'][f'rolling_mean_{window}'] * (window-1) + new_value
            ) / window
        
        # Update trend
        state['trend'] += 1
        state['date'] = current_date

class AutoTS:
    def __init__(self, time_budget=600, ensemble_size=4):
        self.models = {
            'prophet': None,
            'arima': None,
            'xgboost': None,
            'automl': None
        }
        self.ensemble_weights = {}
        self.time_budget = time_budget
        self.ensemble_size = ensemble_size
        self.feature_columns = []
        
    def train(self, train_data):
        """Train ensemble of models with automated weighting"""
        try:
            # 1. Train Prophet
            self.models['prophet'] = EnhancedProphet().fit(train_data)
            
            # 2. Train ARIMA
            arima_result = auto_arima_enhanced(train_data['y'])
            self.models['arima'] = arima_result['model']
            
            # 3. Train XGBoost
            self.models['xgboost'] = TemporalXGBoost().fit(train_data)
            
            # 4. Train AutoML
            automl = AutoML()
            X = self._create_automl_features(train_data)
            automl.fit(X, train_data['y'], task='regression', time_budget=self.time_budget)
            self.models['automl'] = automl
            
            # Calculate ensemble weights
            self._calculate_weights(train_data)
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            
        return self
    
    def _create_automl_features(self, data):
        """Feature engineering for AutoML"""
        df = data.copy()
        df['month'] = df['ds'].dt.month
        df['year'] = df['ds'].dt.year
        df['dayofweek'] = df['ds'].dt.dayofweek
        df['quarter'] = df['ds'].dt.quarter
        self.feature_columns = [col for col in df.columns if col not in ['ds', 'y']]
        return df[self.feature_columns]
    
    def _calculate_weights(self, data):
        """Dynamic weighting based on rolling window performance"""
        tscv = TimeSeriesSplit(n_splits=3)
        scores = {model: [] for model in self.models.keys()}
        
        for train_idx, test_idx in tscv.split(data):
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]
            
            for model_name in self.models.keys():
                try:
                    if model_name == 'prophet':
                        m = EnhancedProphet().fit(train)
                        future = m.make_future_dataframe(periods=len(test))
                        forecast = m.predict(future).tail(len(test))['yhat']
                    elif model_name == 'arima':
                        model = auto_arima(train['y'], suppress_warnings=True)
                        forecast = model.predict(n_periods=len(test))
                    elif model_name == 'xgboost':
                        model = TemporalXGBoost().fit(train)
                        forecast = model.predict(train['ds'].iloc[-1], len(test))['yhat']
                    elif model_name == 'automl':
                        X_test = self._create_automl_features(test)
                        forecast = self.models['automl'].predict(X_test)
                        
                    rmse = np.sqrt(mean_squared_error(test['y'], forecast))
                    scores[model_name].append(rmse)
                except:
                    scores[model_name].append(np.inf)
        
        # Calculate weights inversely proportional to RMSE
        avg_scores = {k: np.mean(v) for k, v in scores.items()}
        total = sum(1/s for s in avg_scores.values() if s > 0)
        self.ensemble_weights = {k: (1/v)/total if v > 0 else 0 for k, v in avg_scores.items()}
        
    def predict(self, horizon):
        """Generate ensemble forecast"""
        forecasts = []
        last_date = pd.to_datetime(self.models['prophet'].history_dates[-1])
        
        # Generate individual forecasts
        prophet_forecast = self.models['prophet'].make_future_dataframe(horizon)
        arima_forecast = self.models['arima'].predict(horizon)
        xgb_forecast = self.models['xgboost'].predict(last_date, horizon)
        automl_forecast = self.models['automl'].predict(
            self._create_automl_features(
                pd.DataFrame({'ds': pd.date_range(last_date, periods=horizon)})
            )
        )
        
        # Combine forecasts
        ensemble = (
            prophet_forecast['yhat'] * self.ensemble_weights['prophet'] +
            arima_forecast * self.ensemble_weights['arima'] +
            xgb_forecast['yhat'] * self.ensemble_weights['xgboost'] +
            automl_forecast * self.ensemble_weights['automl']
        )
        
        return pd.DataFrame({
            'ds': pd.date_range(last_date, periods=horizon),
            'yhat': ensemble,
            'yhat_lower': ensemble * 0.9,
            'yhat_upper': ensemble * 1.1
        })

# ====================== ENHANCED ADJUSTMENT FUNCTIONS ======================
def adjust_forecast_by_category(forecast_df, category_scenarios, category_columns):
    """Enhanced category adjustment with dynamic column handling"""
    if not category_columns or not category_scenarios:
        return forecast_df

    # Get valid category columns present in both scenarios and dataframe
    valid_categories = [col for col in category_columns if col in forecast_df.columns]
    
    # Apply adjustments per category group
    adjusted_groups = []
    for cols, group in forecast_df.groupby(valid_categories):
        group = group.copy()
        for cat_col in valid_categories:
            cat_value = cols[valid_categories.index(cat_col)] if isinstance(cols, tuple) else cols
            scenario = category_scenarios.get(cat_col, {}).get(cat_value, {})
            
            if scenario:
                adjustment = scenario.get("adjustment", 0)
                start_date = pd.to_datetime(scenario.get("start_date"))
                end_date = pd.to_datetime(scenario.get("end_date"))
                
                # Apply time-bound adjustments
                mask = (group["ds"] >= start_date) & (group["ds"] <= end_date)
                if mask.any():
                    for col in ["yhat", "yhat_lower", "yhat_upper"]:
                        if col in group.columns:
                            group.loc[mask, col] *= (1 + adjustment / 100)
        
        adjusted_groups.append(group)
    
    # Re-aggregate while preserving category structure
    return pd.concat(adjusted_groups).sort_values("ds").reset_index(drop=True)

def adjust_forecast(forecast_df, demand_shock, seasonality_adjustment, 
                   external_shock, category_scenarios=None, category_columns=None):
    """Updated with proper category column handling"""
    # ... (keep existing global adjustment code) ...
    
    # Modified category adjustment call
    if category_scenarios and category_columns:
        forecast_df = adjust_forecast_by_category(
            forecast_df, 
            category_scenarios,
            category_columns
        )
    
    return forecast_df

# ====================== ENHANCED METRICS & VISUALIZATION ======================
def mase(y_true, y_pred, seasonal_period=1):
    """Mean Absolute Scaled Error"""
    naive_error = np.mean(np.abs(np.diff(y_true, seasonal_period)))
    forecast_error = np.mean(np.abs(y_true - y_pred))
    return forecast_error / naive_error

def comprehensive_metrics(y_true, y_pred):
    """Enhanced evaluation metrics with proper imports"""
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred)/np.maximum(y_true, 1e-8))) * 100,  # Handle zero division
        'R2': r2_score(y_true, y_pred)
    }
    
    try:
        metrics['MASE'] = mase(y_true, y_pred, m=12)
    except Exception as e:
        metrics['MASE'] = np.nan
        st.warning(f"MASE calculation failed: {str(e)}")
    
    try:
        dir_acc = np.mean(np.sign(y_true[1:]-y_true[:-1]) == np.sign(y_pred[1:]-y_pred[:-1])) * 100
        metrics['DirectionalAccuracy'] = dir_acc
    except:
        metrics['DirectionalAccuracy'] = np.nan
    
    return metrics

def plot_stationarity_analysis(data, y_original, stationarity, diff_order):
    """Enhanced visualization of stationarity analysis"""
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Original Series", "Transformed Series"))
    
    # Original series
    fig.add_trace(go.Scatter(
        x=y_original["ds"], y=y_original["y_original"],
        mode="lines", name="Original", line=dict(color="blue")),
        row=1, col=1
    )
    
    # Transformed series
    if diff_order > 0:
        fig.add_trace(go.Scatter(
            x=data["ds"], y=data["y"],
            mode="lines", name=f"Differenced (Order {diff_order})", line=dict(color="orange")),
            row=2, col=1
        )
    else:
        fig.add_trace(go.Scatter(
            x=data["ds"], y=data["y"],
            mode="lines", name="Original", line=dict(color="green")),
            row=2, col=1
        )
    
    # Annotation
    stationarity_text = "<br>".join([f"{k}: {v['stationarity']}" for k, v in stationarity.items()])
    fig.update_layout(
        title=f"Stationarity Analysis (Global: {stationarity['global']['stationarity']})",
        annotations=[
            dict(
                x=0.5, y=-0.2,
                xref="paper", yref="paper",
                text=stationarity_text,
                showarrow=False
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

def shape_score(actual, forecast):
    """Calculate directional correlation with error handling"""
    try:
        if len(actual) != len(forecast):
            min_len = min(len(actual), len(forecast))
            actual = actual[:min_len]
            forecast = forecast[:min_len]
        
        # Handle constant values case
        if np.std(actual) == 0 or np.std(forecast) == 0:
            return 0.0
            
        corr, _ = pearsonr(actual, forecast)
        return corr
    except:
        return np.nan

def combined_score(rmse, corr, max_rmse, alpha=0.5, beta=1.0):
    """Enhanced combined metric with safety checks"""
    try:
        # Prevent division by zero
        max_rmse = max(max_rmse, 1e-8)
        # Normalize correlation to [0, 2] range
        norm_corr = (1 - np.clip(corr, -1, 1))
        # Normalize RMSE to [0, 1]
        norm_rmse = rmse / max_rmse
        return alpha * norm_rmse + beta * norm_corr
    except:
        return np.nan

# ====================== PROPHET MODEL ======================
def train_prophet_model(train, test, forecast_period, best_params, last_historical_value,
                        is_diff, demand_shock, seasonality_adjustment, external_shock, 
                        category_scenarios=None):
    """Updated Prophet training with enhanced parameter filtering and custom seasonality handling"""
    result = {}
    try:
        # 1. Filter valid Prophet parameters
        valid_prophet_params = [
            'growth', 'changepoint_prior_scale', 'seasonality_prior_scale',
            'seasonality_mode', 'yearly_seasonality', 'weekly_seasonality',
            'daily_seasonality', 'holidays', 'holidays_prior_scale',
            'changepoint_range', 'mcmc_samples', 'uncertainty_samples'
        ]
        
        prophet_params = {
            k: v for k, v in best_params.items() 
            if k in valid_prophet_params
        }

        # 2. Handle custom seasonalities separately
        custom_seasons = best_params.get('custom_seasonalities', [])
        
        # 3. Initialize model with valid parameters
        model = EnhancedProphet(**prophet_params)
        
        # 4. Add custom seasonalities before detection
        if custom_seasons:
            for season in custom_seasons:
                try:
                    model.add_seasonality(
                        name=season['name'],
                        period=season['period'],
                        fourier_order=season['fourier_order']
                    )
                except Exception as e:
                    st.warning(f"Couldn't add {season['name']}: {str(e)}")

        # 5. Automatic seasonality detection
        model.detect_seasonalities(train)

        # 6. Handle growth requirements
        growth_mode = prophet_params.get('growth', 'linear')
        if growth_mode == 'logistic':
            train["cap"] = 1.2 * train["y"].max()
            train["floor"] = 1

        # 7. Fit model with progress tracking
        with st.spinner("🔮 Training Prophet model..."):
            model.fit(train)

        # 8. Create future dataframe
        future = model.make_future_dataframe(
            periods=forecast_period, 
            freq="MS", 
            include_history=False
        )
        
        if growth_mode == 'logistic':
            future["cap"] = train["cap"].max()
            future["floor"] = 1

        # 9. Generate predictions
        forecast = model.predict(future)
        forecast = forecast[forecast["ds"] > train["ds"].max()]

        # 10. Apply scenario adjustments
        forecast = adjust_forecast(
            forecast, 
            demand_shock, 
            seasonality_adjustment, 
            external_shock, 
            category_scenarios
        )

        # 11. Handle differencing
        if is_diff and last_historical_value is not None:
            forecast = inverse_difference(forecast, last_historical_value)

        # 12. Calculate metrics with alignment check
        match_len = min(len(test["y"]), len(forecast))
        if match_len > 0:
            rmse = mean_squared_error(
                test["y"].iloc[:match_len], 
                forecast["yhat"].iloc[:match_len], 
                squared=False
            )
            mape = mean_absolute_percentage_error(
                test["y"].iloc[:match_len],
                forecast["yhat"].iloc[:match_len]
            )
        else:
            rmse, mape = float('nan'), float('nan')

        result = {
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "Forecast": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            "Seasonalities": model.custom_seasonalities,
            "ModelParams": prophet_params,
            "GrowthMode": growth_mode
        }

    except Exception as e:
        st.error(f"Prophet model failed: {str(e)}")

    return "Prophet", result

# ====================== ARIMA MODEL ======================
def train_arima_model(train, test, forecast_period, last_historical_value, 
                      is_diff, demand_shock, seasonality_adjustment,
                      external_shock, category_scenarios=None):
    result = {}
    try:
        if category_scenarios:
            train = train.groupby("ds", as_index=False).agg({"y": "sum"})

        # Enhanced differencing handling
        diff_order = 0
        if is_diff:
            train, diff_order = dynamic_differencing(train, "y")

        # Train with explainability
        model, explanation = auto_arima_enhanced(
            train.set_index("ds")["y"],
            diff_order=diff_order
        )
        
        # Generate predictions
        preds, conf_int = model.predict(
            n_periods=forecast_period,
            return_conf_int=True
        )
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            "ds": pd.date_range(
                start=train["ds"].iloc[-1] + pd.DateOffset(months=1),
                periods=forecast_period,
                freq="MS"
            ),
            "yhat": preds,
            "yhat_lower": conf_int[:, 0],
            "yhat_upper": conf_int[:, 1]
        })
        
        # Apply scenarios
        forecast_df = adjust_forecast(forecast_df, demand_shock, 
                                    seasonality_adjustment, external_shock,
                                    category_scenarios)
        
        # Differencing reversal
        if is_diff and last_historical_value is not None:
            forecast_df = inverse_difference(forecast_df, last_historical_value)
        
        # Enhanced evaluation
        metrics = comprehensive_metrics(
            test["y"].iloc[:len(forecast_df)],
            forecast_df["yhat"]
        )
        
        result = {
            **metrics,
            "Forecast": forecast_df,
            "Model": model,
            "Explanation": explanation
        }

    except Exception as e:
        st.error(f"ARIMA Error: {str(e)}")
        result = {"error": str(e)}
    
    return "ARIMA", result

# ====================== XGBOOST MODEL ======================
def train_xgb_model(train, test, forecast_period, last_historical_value,
                    is_diff, demand_shock, seasonality_adjustment,
                    external_shock, category_scenarios=None):
    result = {}
    try:
        if category_scenarios:
            train = train.groupby("ds", as_index=False).agg({"y": "sum"})

        # Initialize and train model
        model = TemporalXGBoost(horizon=forecast_period)
        model.fit(train)
        
        # Generate forecasts
        start_date = train["ds"].iloc[-1]
        forecast_df = model.predict(start_date, forecast_period)
        
        # Apply scenario adjustments
        forecast_df = adjust_forecast(
            forecast_df, 
            demand_shock,
            seasonality_adjustment,
            external_shock,
            category_scenarios
        )
        
        # Handle differencing
        if is_diff and last_historical_value is not None:
            forecast_df["yhat"] = inverse_difference(
                forecast_df["yhat"], 
                last_historical_value
            )
        
        # Calculate metrics
        match_len = min(len(test["y"]), len(forecast_df))
        metrics = {
            "RMSE": mean_squared_error(
                test["y"].iloc[:match_len], 
                forecast_df["yhat"].iloc[:match_len], 
                squared=False
            ),
            "MAPE": mean_absolute_percentage_error(
                test["y"].iloc[:match_len],
                forecast_df["yhat"].iloc[:match_len]
            ),
            "FeatureImportances": dict(zip(
                model.feature_columns,
                model.model.feature_importances_
            ))
        }
        
        result = {
            **metrics,
            "Forecast": forecast_df,
            "Model": model
        }

    except Exception as e:
        st.error(f"XGBoost Error: {str(e)}")
        result = {"error": str(e)}
    
    return "XGBoost", result

# ====================== AUTOML MODEL ======================
def train_automl_model(train, test, forecast_period, last_historical_value,
                       is_diff, demand_shock, seasonality_adjustment,
                       external_shock, category_scenarios=None, 
                       time_budget=600):
    result = {}
    try:
        # Initialize AutoTS system
        automl = AutoTS(time_budget=time_budget)
        
        # Train ensemble
        automl.train(train)
        
        # Generate future dataframe
        future_dates = pd.date_range(
            start=train["ds"].iloc[-1] + pd.DateOffset(months=1),
            periods=forecast_period,
            freq="MS"
        ).to_frame(index=False, name="ds")
        
        # Generate predictions
        ensemble_pred = automl.predict(future_dates)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            "ds": future_dates["ds"],
            "yhat": ensemble_pred,
            "yhat_lower": ensemble_pred * 0.9,
            "yhat_upper": ensemble_pred * 1.1
        })
        
        # Apply scenarios and differencing
        forecast_df = adjust_forecast(forecast_df, demand_shock,
                                    seasonality_adjustment, external_shock,
                                    category_scenarios)
        
        if is_diff and last_historical_value is not None:
            forecast_df = inverse_difference(forecast_df, last_historical_value)
        
        # Enhanced evaluation
        metrics = comprehensive_metrics(
            test["y"].iloc[:len(forecast_df)],
            forecast_df["yhat"]
        )
        
        result = {
            **metrics,
            "Forecast": forecast_df,
            "EnsembleWeights": automl.ensemble_weights,
            "Models": automl.models
        }

    except Exception as e:
        st.error(f"AutoML Error: {str(e)}")
        result = {"error": str(e)}
    
    return "AutoML", result

def main():
    # Set subscription level: "free" for basic features, "premium" for full access
    user_id = "user123"
    subscription_level = "premium"  # Change to "premium" to enable advanced features

    if subscription_level != "premium":
        st.info("You are using the Free version. Advanced features such as category adjustments, extended forecast horizons, hyperparameter tuning, and forecast downloads are disabled.")

    if st.button("🔄 Reset App"):
        st.session_state.clear()
        st.experimental_rerun()

    uploaded_file = st.file_uploader("Upload your sales data file", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(data)

            st.markdown(
                """
                <div style="text-align: center;">
                    <h2 style="color: #2B3A42;">🛠️ Map Your Columns</h2>
                </div>
                """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                date_column = st.selectbox(
                    "📅 Select the Date Column:",
                    ["-- Select Column --"] + list(data.columns),
                    key="date_col"
                )
            with col2:
                sales_column = st.selectbox(
                    "💰 Select the Sales Column:",
                    ["-- Select Column --"] + list(data.columns),
                    key="sales_col"
                )
            with col3:
                if date_column == "-- Select Column --" or sales_column == "-- Select Column --":
                    st.warning("Please select the Date and Sales columns first.")
                    category_columns = None
                else:
                    if subscription_level != "premium":
                        st.info("Category adjustments are available only for premium users.")
                        category_columns = None
                    else:
                        category_columns = st.multiselect(
                            "🏷️ Select Category Columns (Optional):",
                            options=[col for col in data.columns if col not in [date_column, sales_column]],
                            key="category_cols"
                        )

            if date_column != "-- Select Column --":
                data[date_column] = pd.to_datetime(data[date_column], errors="coerce")

            if subscription_level != "premium":
                time_budget = 60
                forecast_period = 3
            else:
                st.markdown("### ⏱️ AutoML Time Budget")
                time_budget = st.slider(
                    "Set the time budget for AutoML training (in seconds):",
                    min_value=60, max_value=1200, value=300, step=60,
                    help="Increase the time budget for larger datasets or more complex models."
                )
                forecast_period = 24

            if date_column != "-- Select Column --" and sales_column != "-- Select Column --":
                last_date_in_data = data[date_column].max()
                min_future_date = (last_date_in_data + pd.DateOffset(days=1)).date()

                if subscription_level != "premium":
                    st.sidebar.info("Scenario planning is available only for premium users.")
                    demand_shock = 0
                    seasonality_adjustment = 0
                    external_shock = False
                    category_scenarios = {}
                else:
                    st.sidebar.markdown("### 🎯 Scenario Planning")
                    demand_shock = st.sidebar.slider(
                        "Simulate Demand Shock (% Change in Sales):",
                        min_value=-50, max_value=50, value=0, step=5
                    )
                    seasonality_adjustment = st.sidebar.slider(
                        "Adjust Seasonality Strength (% Change):",
                        min_value=-50, max_value=50, value=0, step=5
                    )
                    external_shock = st.sidebar.checkbox("Simulate External Shock (e.g., Economic Downturn)")
                    category_scenarios = {}
                    if category_columns:
                        st.sidebar.markdown("### 🎯 Scenario Planning by Category (Dynamic)")
                        for col in category_columns:
                            st.sidebar.markdown(f"#### Adjustments for '{col}'")
                            unique_cats = sorted(data[col].dropna().unique())
                            selected_cats = st.sidebar.multiselect(
                                f"Pick categories in '{col}' to adjust:",
                                options=unique_cats,
                                help=f"Select one or more categories from '{col}' that you want to adjust."
                            )
                            category_scenarios[col] = {}
                            for cat in selected_cats:
                                with st.sidebar.expander(f"Adjust '{cat}' in '{col}'"):
                                    cat_adjust = st.slider(
                                        f"Percentage change for '{cat}'",
                                        min_value=-50, max_value=50, value=0, step=5,
                                        help=f"Adjust sales for category '{cat}' within column '{col}'"
                                    )
                                    cat_start = st.date_input(
                                        f"Start date for '{cat}'",
                                        value=min_future_date,
                                        min_value=min_future_date
                                    )
                                    default_end = (last_date_in_data + pd.DateOffset(months=1)).date()
                                    cat_end = st.date_input(
                                        f"End date for '{cat}'",
                                        value=default_end if default_end > min_future_date else min_future_date,
                                        min_value=cat_start
                                    )
                                    category_scenarios[col][cat] = {
                                        "adjustment": cat_adjust,
                                        "start_date": cat_start,
                                        "end_date": cat_end
                                    }
                    else:
                        st.sidebar.warning("Please select the Date and Sales columns to enable scenario planning.")
                        demand_shock = 0
                        seasonality_adjustment = 0
                        external_shock = False
                        category_scenarios = {}

            if date_column != "-- Select Column --" and sales_column != "-- Select Column --":
                start_forecast = st.button("✅ Start Forecast", key="start_btn",
                                           help="Click to generate your AI-powered forecast")
            else:
                start_forecast = st.button("⏳ Select Columns First", disabled=True, key="start_disabled")

            if subscription_level == "premium":
                total_steps = 12
            else:
                total_steps = 6

            overall_status = st.empty()
            prophet_status = st.empty()
            arima_status = st.empty()
            xgb_status = st.empty()
            automl_status = st.empty()
            progress_bar = st.progress(0)
            step_message = st.empty()

            if start_forecast:
                # STEP 1: Preprocess Data
                step = 1
                step_message.text(f"Step {step} of {total_steps}: Preprocessing data...")
                with st.spinner("🔍 Preprocessing data..."):
                    processed_data, last_historical_value, y_original, is_diff = preprocess_data(
                        data, date_column, sales_column, category_columns
                    )
                    time.sleep(1)
                if processed_data is None:
                    st.error("Preprocessing failed. Please check your data.")
                    return
                st.success("✅ Data Preprocessed Successfully!")

                last_historical_date = y_original["ds"].max()
                # overall_status.info(f"🔍 Last Historical Date: {last_historical_date}")
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(0.5)

                # STEP 2: Review Processed Data
                step += 1
                step_message.text(f"Step {step} of {total_steps}: Reviewing processed data...")
                st.markdown(
                    """
                    <div style="text-align: center;">
                        <h2 style="color: #2B3A42;">📅 Preprocessed Monthly Data</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with st.expander("📊 View Processed Data"):
                    st.dataframe(processed_data.style.set_properties(**{"text-align": "center"}), width=1400, height=450)
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(0.5)

                # STEP 3: Split Data into Training and Test Sets
                step += 1
                step_message.text(f"Step {step} of {total_steps}: Splitting data into training and test sets...")
                testing_period = int(len(processed_data) * 0.2)
                train = processed_data.iloc[:-testing_period]
                test = processed_data.iloc[-testing_period:]
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(1)

                if subscription_level == "premium":
                    # PREMIUM PIPELINE
                    # STEP 4: Tune Prophet Model with Optuna
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Tuning Prophet model with Optuna...")
                    with st.spinner("🚀 Running advanced hyperparameter optimization..."):
                        try:
                            tune_result = tune_prophet(train, n_trials=50)
                            best_params = {
                                **tune_result['params'],
                                'custom_seasonalities': tune_result['custom_seasonalities']
                            }
                            best_rmse = tune_result['metrics']['best_rmse']
                            time.sleep(1)
                        except Exception as e:
                            st.error(f"Hyperparameter tuning failed: {str(e)}")
                            return
                    
                    st.success(f"✅ Best Prophet Params: {best_params}")
                    overall_status.write(f"📉 Best RMSE (CV): {best_rmse:.2f}")

                        # Add optimization visualization
                    with st.expander("🔍 View Hyperparameter Optimization Results"):
                        try:
                            from optuna.visualization import plot_optimization_history
                            fig = plot_optimization_history(tune_result['study'])
                            st.plotly_chart(fig)
                        except Exception as e:
                            st.warning(f"Could not display optimization details: {str(e)}")

                    # STEP 5: Train Prophet Model
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Training Prophet model...")
                    with st.spinner("🚀 Training Prophet model..."):
                        prophet_model_name, prophet_res = train_prophet_model(
                            train, test, forecast_period, best_params,
                            last_historical_value, is_diff,
                            demand_shock, seasonality_adjustment, external_shock, category_scenarios
                        )
                        time.sleep(1)
                    st.success("✅ Prophet Model Training Complete!")
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)

                    # # STEP 6: Train ARIMA Model
                    # step += 1
                    # step_message.text(f"Step {step} of {total_steps}: Training ARIMA model...")
                    # with st.spinner("🚀 Training ARIMA model..."):
                    #     arima_model_name, arima_res = train_arima_model(
                    #         train, test, forecast_period,
                    #         last_historical_value, is_diff,
                    #         demand_shock, seasonality_adjustment, external_shock, category_scenarios
                    #     )
                    #     time.sleep(1)
                    # st.success("✅ ARIMA Model Training Complete!")
                    # progress_bar.progress(int((step / total_steps) * 100))
                    # time.sleep(1)

                    # STEP 7: Train XGBoost Model
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Training XGBoost model...")
                    with st.spinner("🚀 Training XGBoost model..."):
                        xgb_model_name, xgb_res = train_xgb_model(
                            train, test, forecast_period,
                            last_historical_value, is_diff,
                            demand_shock, seasonality_adjustment, external_shock, category_scenarios
                        )
                        time.sleep(1)
                    st.success("✅ XGBoost Model Training Complete!")
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)

                    # STEP 9: Train AutoML Model
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Training AutoML model...")
                    with st.spinner("🚀 Training AutoML Model..."):
                        automl_model_name, automl_res = train_automl_model(
                            train, test, forecast_period,
                            last_historical_value, is_diff,
                            demand_shock, seasonality_adjustment, external_shock,
                            category_scenarios, time_budget
                        )
                        time.sleep(1)
                    st.success("✅ AutoML Model Training Complete!")
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)

                    # STEP 9: Compile Forecast Results (Premium)
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Compiling forecast results...")
                    st.success("🎉 Forecasting process completed!")
                    time.sleep(1)
                    results = {
                        prophet_model_name: prophet_res,
                        # arima_model_name: arima_res,
                        xgb_model_name: xgb_res,
                        automl_model_name: automl_res
                    }
                    valid_results = {model: res for model, res in results.items() if res.get("Forecast") is not None}
                    if not valid_results:
                        st.error("No valid model forecasts produced.")
                        return
                    st.session_state.model_results = valid_results
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)

                    # STEP 10: Display Model Performance Comparison (Premium)
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Displaying model performance comparison...")
                    comparison_data = []

                    with st.expander("📊 Model Performance Metrics", expanded=True):
                        if not st.session_state.model_results:
                            st.warning("No model results found. Please train the models first.")
                        else:
                            # Calculate metrics with error handling
                            valid_rmses = []
                            for model, res in st.session_state.model_results.items():
                                try:
                                    forecast_df = res["Forecast"]
                                    match_len = min(len(test["y"]), len(forecast_df))
                                    actual = test["y"].iloc[:match_len].values
                                    pred = forecast_df["yhat"].iloc[:match_len].values
                                    
                                    # Calculate metrics with NaN protection
                                    rmse = np.nan
                                    mape = np.nan
                                    corr = np.nan
                                    
                                    if len(actual) > 0 and len(pred) > 0:
                                        with warnings.catch_warnings():
                                            warnings.simplefilter("ignore")
                                            rmse = np.sqrt(mean_squared_error(actual, pred))
                                            mape = np.mean(np.abs((actual - pred)/np.maximum(actual, 1e-8))) * 100
                                            corr = pearsonr(actual, pred)[0] if np.std(actual) > 0 and np.std(pred) > 0 else np.nan
                                    
                                    res.update({
                                        "RMSE": rmse,
                                        "MAPE": mape,
                                        "Shape (corr)": corr
                                    })
                                    valid_rmses.append(rmse) if not np.isnan(rmse) else None
                                    
                                    comparison_data.append({
                                        "Model": model,
                                        "RMSE": rmse,
                                        "MAPE": mape,
                                        "Shape (corr)": corr
                                    })
                                    
                                except Exception as e:
                                    st.error(f"Error evaluating {model}: {str(e)}")
                                    comparison_data.append({
                                        "Model": model,
                                        "RMSE": np.nan,
                                        "MAPE": np.nan,
                                        "Shape (corr)": np.nan
                                    })

                            # Calculate combined scores
                            max_rmse = max(valid_rmses) if valid_rmses else 1.0
                            for item in comparison_data:
                                item["Combined Score"] = combined_score(
                                    item["RMSE"], 
                                    item["Shape (corr)"], 
                                    max_rmse,
                                    alpha=0.5,
                                    beta=1.0
                                )

                            # Create styled dataframe
                            display_df = pd.DataFrame(comparison_data)
                            
                            # Format numbers and handle NaNs
                            styled_df = display_df.style \
                                .format({
                                    "RMSE": "{:.2f}",
                                    "MAPE": "{:.1%}",
                                    "Shape (corr)": "{:.2f}",
                                    "Combined Score": "{:.3f}"
                                }, na_rep="N/A") \
                                .background_gradient(
                                    subset=["Combined Score"], 
                                    cmap="YlGn",
                                    vmin=display_df["Combined Score"].min(),
                                    vmax=display_df["Combined Score"].max()
                                ) \
                                .set_caption(
                                    "Model Performance Comparison (Lower Combined Score is Better)\n"
                                    "Combined Score = 50% Normalized RMSE + 50% (1 - Correlation)"
                                )

                            # Display results
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Find and display best model
                            if not display_df.empty:
                                best_model = display_df.loc[display_df["Combined Score"].idxmin(), "Model"]
                                st.success(f"✨ **AI-Recommended Model:** `{best_model}`")
                                st.markdown(f"**Rationale:** Selected based on optimal balance between error minimization (RMSE) "
                                            f"and pattern matching (Correlation) metrics")
                                
                                # Show metric definitions
                                with st.expander("📖 Metric Explanations"):
                                    st.markdown("""
                                    - **RMSE (Root Mean Squared Error):** Measures average error magnitude
                                    - **MAPE (Mean Absolute Percentage Error):** Shows average percentage error
                                    - **Shape Correlation:** Measures pattern matching (1 = perfect match)
                                    - **Combined Score:** Balanced metric (50% RMSE, 50% pattern matching)
                                    """)

                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)

                    # 🔮 AI-Powered Future Insights + Category Summary + High-Risk Detection
                    try:
                        # Retrieve the best model's forecast DataFrame
                        forecast_data = st.session_state.model_results[best_model]["Forecast"]

                        # Compute AI-powered insights
                        highest_point = forecast_data.loc[forecast_data["yhat"].idxmax()]
                        lowest_point = forecast_data.loc[forecast_data["yhat"].idxmin()]
                        projected_growth = ((forecast_data["yhat"].iloc[-1] - test["y"].iloc[-1]) / test["y"].iloc[-1]) * 100
                        trend = "📈 **Growth Expected**" if projected_growth > 0 else "📉 **Potential Decline**"

                        insights_text = f"""
                    - **Projected Sales Growth:** {abs(projected_growth):.2f}% {trend}
                    - **Peak Sales Expected:** ${highest_point['yhat']:.2f} on {highest_point['ds'].strftime('%Y-%m-%d')}
                    - **Lowest Predicted Sales:** ${lowest_point['yhat']:.2f} on {lowest_point['ds'].strftime('%Y-%m-%d')}
                    - **Optimal Decision Window:** Plan around peak sales in {highest_point['ds'].strftime('%B %Y')}
                    - **Risk Zones Identified:** Check months marked as 🔥 'High-Risk' below
                    - **Volatility Analysis:** Forecast suggests a {'stable' if abs(projected_growth) < 5 else 'fluctuating'} trend
                        """

                        with st.expander("🔮 AI-Powered Future Insights", expanded=True):
                            st.markdown(insights_text)

                        # Build a summary of category adjustments if any were applied.
                        if category_scenarios:
                            cat_adj_summary = "### Category Adjustments Summary\n"
                            for col, adjustments in category_scenarios.items():
                                cat_adj_summary += f"- **{col}**:\n"
                                for cat, details in adjustments.items():
                                    cat_adj_summary += (
                                        f"  - **{cat}**: {details['adjustment']}% adjustment "
                                        f"from {details['start_date']} to {details['end_date']}\n"
                                    )
                            st.markdown(cat_adj_summary)

                        # 🔥 Detect High-Risk Periods in Forecast
                        if forecast_data is not None:
                            try:
                                forecast_data["volatility"] = forecast_data["yhat"].rolling(3).std()
                                forecast_data["risk"] = "✅ Stable"

                                # Define thresholds at 75th and 90th percentile
                                p75 = forecast_data["volatility"].quantile(0.75)
                                p90 = forecast_data["volatility"].quantile(0.90)

                                forecast_data.loc[forecast_data["volatility"] > p75, "risk"] = "⚠️ High Volatility"
                                forecast_data.loc[forecast_data["volatility"] > p90, "risk"] = "❌ Major Decline"

                                st.markdown("### 🚨 High-Risk Sales Periods Identified")
                                st.dataframe(
                                    forecast_data[["ds", "yhat", "volatility", "risk"]]
                                    .style.applymap(
                                        lambda x: (
                                            "background-color: #FFDDC1" if x == "❌ Major Decline" else
                                            "background-color: #FFEEAA" if x == "⚠️ High Volatility" else
                                            "background-color: #C6ECAE"
                                        ),
                                        subset=["risk"]
                                    )
                                )
                            except Exception as e:
                                st.error(f"❌ Error detecting high-risk periods: {e}")

                    except Exception as e:
                        st.error(f"❌ Error analyzing forecast data: {e}")

                    # STEP 11: Finalize Forecast Visualization (Premium)
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Finalizing forecast visualization...")
                    st.markdown("### 🔍 Forecast Comparison Across Models")
                    model_colors = {
                        "Prophet": "blue",
                        # "ARIMA": "green",
                        "XGBoost": "red",
                        "AutoML": "purple"
                    }
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_original["ds"],
                        y=y_original["y_original"],
                        mode="lines",
                        name="Historical Data",
                        line=dict(color="black", width=2)
                    ))
                    for model_name, res in results.items():
                        forecast_df = res["Forecast"]
                        fig.add_trace(go.Scatter(
                            x=forecast_df["ds"],
                            y=forecast_df["yhat"],
                            mode="lines",
                            name=f"{model_name} Forecast",
                            line=dict(width=2, color=model_colors.get(model_name, "gray"))
                        ))
                    fig.update_layout(
                        title="📊 Multi-Model Sales Forecast",
                        xaxis_title="Date",
                        yaxis_title="Sales",
                        legend_title="Models",
                        template="plotly_white",
                        xaxis_tickformat="%Y-%m"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    progress_bar.progress(100)
                    step_message.text("All steps completed!")
                    
                    st.markdown("### 📥 Download Forecast Data")
                    try:
                        csv = st.session_state.model_results[best_model]["Forecast"].to_csv(index=False)
                        st.download_button(
                            label="📩 Download Best Model Forecast (CSV)",
                            data=csv,
                            file_name="forecast.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"❌ Error generating download file: {e}")
                else:
                    # FREE USERS PIPELINE (only AutoML)
                    # STEP 4 (Free): Train AutoML Model
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Training AutoML model (Limited)...")
                    with st.spinner("🚀 Training AutoML Model..."):
                        automl_model_name, automl_res = train_automl_model(
                            train, test, forecast_period,
                            last_historical_value, is_diff,
                            time_budget, demand_shock, seasonality_adjustment, external_shock, category_scenarios
                        )
                        time.sleep(1)
                    if is_diff and automl_res.get("Forecast") is not None:
                        automl_res["Forecast"] = inverse_difference(automl_res["Forecast"], last_historical_value)
                    automl_status.success("✅ AutoML Model Training Complete!")
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)
                    
                    # STEP 5 (Free): Compile Forecast Results
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Compiling forecast results...")
                    st.success("🎉 Forecasting process completed!")
                    time.sleep(1)
                    results = {"AutoML": automl_res}
                    st.session_state.model_results = results
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)
                    
                    # STEP 6 (Free): Finalize Forecast Visualization
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Finalizing forecast visualization...")
                    st.markdown("### 🔍 Forecast Visualization")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_original["ds"],
                        y=y_original["y_original"],
                        mode="lines",
                        name="Historical Data",
                        line=dict(color="black", width=2)
                    ))
                    forecast_df = automl_res["Forecast"]
                    fig.add_trace(go.Scatter(
                        x=forecast_df["ds"],
                        y=forecast_df["yhat"],
                        mode="lines",
                        name="AutoML Forecast",
                        line=dict(width=2, color="purple")
                    ))
                    fig.update_layout(
                        title="📊 Sales Forecast",
                        xaxis_title="Date",
                        yaxis_title="Sales",
                        template="plotly_white",
                        xaxis_tickformat="%Y-%m"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    progress_bar.progress(100)
                    step_message.text("All steps completed!")
                    
                    st.info("Upgrade to Premium to unlock advanced features like multi-model comparison and forecast download.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()