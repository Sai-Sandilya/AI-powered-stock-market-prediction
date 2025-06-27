"""
Advanced Ensemble System
Combines multiple model architectures with weighted voting, confidence-based stacking,
and sophisticated model selection for superior prediction performance.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class AdvancedEnsemble:
    def __init__(self, sequence_length: int = 60, n_features: int = 50):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.models = {}
        self.scalers = {}
        self.model_weights = {}
        self.confidence_scores = {}
        self.performance_history = {}
        
    def create_lstm_architectures(self) -> Dict[str, tf.keras.Model]:
        """
        Create multiple LSTM architectures with different designs.
        """
        architectures = {}
        
        # 1. Standard LSTM
        model1 = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model1.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        architectures['lstm_standard'] = model1
        
        # 2. Bidirectional LSTM
        model2 = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.3),
            Bidirectional(LSTM(50, return_sequences=False)),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model2.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        architectures['lstm_bidirectional'] = model2
        
        # 3. Deep LSTM
        model3 = Sequential([
            LSTM(150, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.3),
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            LSTM(50, return_sequences=False),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model3.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        architectures['lstm_deep'] = model3
        
        # 4. CNN-LSTM Hybrid
        model4 = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.sequence_length, self.n_features)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(50, return_sequences=False),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model4.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        architectures['cnn_lstm'] = model4
        
        # 5. Attention LSTM
        inputs = Input(shape=(self.sequence_length, self.n_features))
        lstm1 = LSTM(100, return_sequences=True)(inputs)
        lstm2 = LSTM(50, return_sequences=True)(lstm1)
        
        # Simple attention mechanism
        attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=50)(lstm2, lstm2)
        attention_output = tf.keras.layers.LayerNormalization()(attention + lstm2)
        
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
        dense1 = Dense(25, activation='relu')(pooled)
        dropout = Dropout(0.3)(dense1)
        outputs = Dense(1)(dropout)
        
        model5 = Model(inputs=inputs, outputs=outputs)
        model5.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        architectures['lstm_attention'] = model5
        
        return architectures
    
    def create_ml_models(self) -> Dict[str, object]:
        """
        Create traditional machine learning models.
        """
        models = {}
        
        # Tree-based models
        models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        # Linear models
        models['ridge'] = Ridge(alpha=1.0, random_state=42)
        models['lasso'] = Lasso(alpha=0.1, random_state=42)
        models['linear'] = LinearRegression()
        
        # Support Vector Regression
        models['svr'] = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        return models
    
    def create_meta_learner(self) -> object:
        """
        Create a meta-learner for stacking.
        """
        return Ridge(alpha=0.5, random_state=42)
    
    def calculate_model_confidence(self, predictions: np.ndarray, actual: np.ndarray) -> float:
        """
        Calculate confidence score based on prediction accuracy and consistency.
        """
        if len(predictions) == 0 or len(actual) == 0:
            return 0.5
        
        # Calculate various confidence metrics
        mse = mean_squared_error(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        
        # Prediction consistency (lower variance = higher confidence)
        pred_variance = np.var(predictions)
        actual_variance = np.var(actual)
        consistency = 1 / (1 + abs(pred_variance - actual_variance))
        
        # Directional accuracy
        actual_direction = np.diff(actual) > 0
        pred_direction = np.diff(predictions) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        # Combine metrics into confidence score
        confidence = (
            0.3 * (1 / (1 + mse)) +  # MSE component
            0.2 * (1 / (1 + mae)) +  # MAE component
            0.3 * consistency +       # Consistency component
            0.2 * directional_accuracy  # Directional accuracy component
        )
        
        return np.clip(confidence, 0, 1)
    
    def adaptive_weighting(self, predictions: Dict[str, np.ndarray], 
                          confidences: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate adaptive weights based on confidence scores and recent performance.
        """
        total_confidence = sum(confidences.values())
        
        if total_confidence == 0:
            # Equal weights if no confidence
            n_models = len(predictions)
            return {name: 1.0/n_models for name in predictions.keys()}
        
        # Base weights from confidence
        base_weights = {name: conf/total_confidence for name, conf in confidences.items()}
        
        # Adjust weights based on recent performance
        adjusted_weights = {}
        for name, base_weight in base_weights.items():
            if name in self.performance_history:
                recent_performance = np.mean(self.performance_history[name][-5:])  # Last 5 periods
                performance_factor = 1 + recent_performance
                adjusted_weights[name] = base_weight * performance_factor
            else:
                adjusted_weights[name] = base_weight
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        normalized_weights = {name: weight/total_weight for name, weight in adjusted_weights.items()}
        
        return normalized_weights
    
    def train_ensemble(self, df: pd.DataFrame, validation_split: float = 0.2):
        """
        Train all models in the advanced ensemble.
        """
        print("🚀 Training advanced ensemble models...")
        
        # Prepare data
        features, targets = self.prepare_data(df)
        
        # Create sequences for LSTM
        X_lstm, y_lstm = self.create_sequences(features, targets)
        
        # Split data
        split_idx = int(len(X_lstm) * (1 - validation_split))
        X_train_lstm, X_val_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train_lstm, y_val_lstm = y_lstm[:split_idx], y_lstm[split_idx:]
        
        # Prepare data for ML models
        X_ml = features[self.sequence_length:split_idx+self.sequence_length]
        y_ml = targets[self.sequence_length:split_idx+self.sequence_length]
        X_val_ml = features[split_idx+self.sequence_length:]
        y_val_ml = targets[split_idx+self.sequence_length:]
        
        # Train LSTM models
        print("📊 Training LSTM architectures...")
        lstm_models = self.create_lstm_architectures()
        
        for name, model in lstm_models.items():
            print(f"   Training {name}...")
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            history = model.fit(
                X_train_lstm, y_train_lstm,
                validation_data=(X_val_lstm, y_val_lstm),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate and store
            val_loss = model.evaluate(X_val_lstm, y_val_lstm, verbose=0)[0]
            self.models[name] = model
            self.performance_history[name] = [val_loss]
            
            print(f"   ✅ {name} - Validation Loss: {val_loss:.4f}")
        
        # Train ML models
        print("🌳 Training ML models...")
        ml_models = self.create_ml_models()
        
        for name, model in ml_models.items():
            print(f"   Training {name}...")
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_ml)
            X_val_scaled = scaler.transform(X_val_ml)
            
            # Train model
            model.fit(X_train_scaled, y_ml)
            
            # Evaluate
            val_pred = model.predict(X_val_scaled)
            val_mse = mean_squared_error(y_val_ml, val_pred)
            
            self.models[name] = model
            self.scalers[name] = scaler
            self.performance_history[name] = [val_mse]
            
            print(f"   ✅ {name} - Validation MSE: {val_mse:.4f}")
        
        print("✅ Advanced ensemble training completed!")
    
    def predict_ensemble(self, df: pd.DataFrame, use_confidence_weighting: bool = True) -> Dict:
        """
        Make predictions using the advanced ensemble with confidence-based weighting.
        """
        print("🔮 Making advanced ensemble predictions...")
        
        # Prepare data
        features, targets = self.prepare_data(df)
        
        # Get predictions from each model
        predictions = {}
        confidences = {}
        
        # LSTM predictions
        if len(features) >= self.sequence_length:
            X_lstm, _ = self.create_sequences(features, targets)
            
            for name, model in self.models.items():
                if name.startswith('lstm_'):
                    pred = model.predict(X_lstm, verbose=0).flatten()
                    predictions[name] = pred
                    
                    # Calculate confidence
                    if len(pred) > 1 and len(targets[self.sequence_length:]) > 1:
                        confidence = self.calculate_model_confidence(
                            pred, targets[self.sequence_length:]
                        )
                    else:
                        confidence = 0.5
                    confidences[name] = confidence
        
        # ML predictions
        X_ml = features[self.sequence_length:]
        
        for name, model in self.models.items():
            if not name.startswith('lstm_'):
                if name in self.scalers:
                    X_scaled = self.scalers[name].transform(X_ml)
                    pred = model.predict(X_scaled)
                    predictions[name] = pred
                    
                    # Calculate confidence
                    if len(pred) > 1 and len(targets[self.sequence_length:]) > 1:
                        confidence = self.calculate_model_confidence(
                            pred, targets[self.sequence_length:]
                        )
                    else:
                        confidence = 0.7  # Default confidence for ML models
                    confidences[name] = confidence
        
        # Calculate ensemble prediction
        if predictions:
            if use_confidence_weighting:
                # Use adaptive weighting
                weights = self.adaptive_weighting(predictions, confidences)
            else:
                # Equal weights
                n_models = len(predictions)
                weights = {name: 1.0/n_models for name in predictions.keys()}
            
            # Weighted ensemble prediction
            ensemble_pred = np.zeros(len(list(predictions.values())[0]))
            for name, pred in predictions.items():
                ensemble_pred += weights[name] * pred
            
            # Calculate ensemble confidence
            ensemble_confidence = np.mean(list(confidences.values()))
            
            # Store weights for analysis
            self.model_weights = weights
            self.confidence_scores = confidences
            
            return {
                'ensemble_prediction': ensemble_pred,
                'ensemble_confidence': ensemble_confidence,
                'individual_predictions': predictions,
                'model_weights': weights,
                'model_confidences': confidences,
                'prediction_std': np.std(list(predictions.values()), axis=0)  # Uncertainty measure
            }
        else:
            return {}
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for different model types.
        """
        # Remove non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_col and col != 'Date']
        
        # Prepare features
        features = df[feature_cols].fillna(0).values
        targets = df[target_col].values
        
        return features, targets
    
    def create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM models.
        """
        X, y = [], []
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(targets[i])
        return np.array(X), np.array(y)
    
    def save_ensemble(self, save_dir: str = 'models'):
        """
        Save all models and metadata.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            if name.startswith('lstm_'):
                model.save(f'{save_dir}/{name}_advanced_ensemble.h5')
            else:
                joblib.dump(model, f'{save_dir}/{name}_advanced_ensemble.pkl')
        
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{save_dir}/{name}_scaler_advanced_ensemble.pkl')
        
        # Save metadata
        metadata = {
            'model_weights': self.model_weights,
            'confidence_scores': self.confidence_scores,
            'performance_history': self.performance_history,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(metadata, f'{save_dir}/advanced_ensemble_metadata.pkl')
        
        print(f"✅ Advanced ensemble models saved to {save_dir}")
    
    def load_ensemble(self, save_dir: str = 'models'):
        """
        Load all models and metadata.
        """
        # Load LSTM models
        lstm_names = ['lstm_standard', 'lstm_bidirectional', 'lstm_deep', 'cnn_lstm', 'lstm_attention']
        
        for name in lstm_names:
            model_path = f'{save_dir}/{name}_advanced_ensemble.h5'
            if os.path.exists(model_path):
                self.models[name] = tf.keras.models.load_model(model_path)
        
        # Load ML models
        ml_names = ['random_forest', 'gradient_boosting', 'ridge', 'lasso', 'linear', 'svr']
        
        for name in ml_names:
            model_path = f'{save_dir}/{name}_advanced_ensemble.pkl'
            scaler_path = f'{save_dir}/{name}_scaler_advanced_ensemble.pkl'
            
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
            if os.path.exists(scaler_path):
                self.scalers[name] = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = f'{save_dir}/advanced_ensemble_metadata.pkl'
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.model_weights = metadata.get('model_weights', {})
            self.confidence_scores = metadata.get('confidence_scores', {})
            self.performance_history = metadata.get('performance_history', {})
        
        print(f"✅ Advanced ensemble models loaded from {save_dir}")

def test_advanced_ensemble():
    """Test the advanced ensemble system."""
    print("🧪 Testing advanced ensemble system...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate synthetic time series data
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    features = np.random.randn(n_samples, n_features)
    targets = np.cumsum(np.random.randn(n_samples) * 0.01) + 100
    
    df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(n_features)])
    df['Date'] = dates
    df['Close'] = targets
    
    # Create and train ensemble
    ensemble = AdvancedEnsemble(sequence_length=30, n_features=n_features)
    ensemble.train_ensemble(df)
    
    # Make predictions
    predictions = ensemble.predict_ensemble(df)
    
    if predictions:
        print(f"📊 Ensemble prediction shape: {predictions['ensemble_prediction'].shape}")
        print(f"🎯 Ensemble confidence: {predictions['ensemble_confidence']:.3f}")
        print(f"⚖️ Model weights: {predictions['model_weights']}")
        print(f"📈 Prediction uncertainty: {np.mean(predictions['prediction_std']):.4f}")
    
    # Save ensemble
    ensemble.save_ensemble()
    
    return ensemble

if __name__ == "__main__":
    test_advanced_ensemble() 