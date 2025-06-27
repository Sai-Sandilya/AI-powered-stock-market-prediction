"""
Ensemble Model Architecture
Combines multiple LSTM models and other architectures for improved prediction accuracy.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten, Input, Concatenate, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class EnsembleModel:
    def __init__(self, sequence_length: int = 60, n_features: int = 50):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def create_lstm_model(self, model_name: str, architecture: str = 'standard') -> tf.keras.Model:
        """
        Create different LSTM architectures.
        """
        if architecture == 'standard':
            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
        elif architecture == 'bidirectional':
            model = Sequential([
                Bidirectional(LSTM(100, return_sequences=True), input_shape=(self.sequence_length, self.n_features)),
                Dropout(0.2),
                Bidirectional(LSTM(50, return_sequences=False)),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
        elif architecture == 'cnn_lstm':
            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.sequence_length, self.n_features)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=32, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
        elif architecture == 'attention_lstm':
            # Input layer
            inputs = Input(shape=(self.sequence_length, self.n_features))
            
            # LSTM layers
            lstm1 = LSTM(100, return_sequences=True)(inputs)
            lstm2 = LSTM(50, return_sequences=True)(lstm1)
            
            # Attention mechanism
            attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=50)(lstm2, lstm2)
            attention_output = tf.keras.layers.LayerNormalization()(attention + lstm2)
            
            # Global average pooling
            pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
            
            # Dense layers
            dense1 = Dense(25, activation='relu')(pooled)
            dropout = Dropout(0.2)(dense1)
            outputs = Dense(1)(dropout)
            
            model = Model(inputs=inputs, outputs=outputs)
            
        elif architecture == 'deep_lstm':
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
                Dropout(0.3),
                LSTM(64, return_sequences=True),
                Dropout(0.3),
                LSTM(32, return_sequences=False),
                Dropout(0.3),
                Dense(50, activation='relu'),
                Dense(25, activation='relu'),
                Dense(1)
            ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def create_ml_models(self):
        """
        Create traditional machine learning models.
        """
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    
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
    
    def train_ensemble(self, df: pd.DataFrame, validation_split: float = 0.2):
        """
        Train all models in the ensemble.
        """
        print("🚀 Training ensemble models...")
        
        # Prepare data
        features, targets = self.prepare_data(df)
        
        # Create sequences for LSTM
        X_lstm, y_lstm = self.create_sequences(features, targets)
        
        # Split data
        split_idx = int(len(X_lstm) * (1 - validation_split))
        X_train_lstm, X_val_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train_lstm, y_val_lstm = y_lstm[:split_idx], y_lstm[split_idx:]
        
        # Prepare data for ML models (use last feature vector from each sequence)
        X_ml = features[self.sequence_length:split_idx+self.sequence_length]
        y_ml = targets[self.sequence_length:split_idx+self.sequence_length]
        X_val_ml = features[split_idx+self.sequence_length:]
        y_val_ml = targets[split_idx+self.sequence_length:]
        
        # Train LSTM models
        lstm_architectures = ['standard', 'bidirectional', 'cnn_lstm', 'attention_lstm', 'deep_lstm']
        
        for arch in lstm_architectures:
            print(f"📊 Training {arch} LSTM...")
            
            # Create and train model
            model = self.create_lstm_model(f"lstm_{arch}", arch)
            
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
            
            # Evaluate model
            val_loss = model.evaluate(X_val_lstm, y_val_lstm, verbose=0)[0]
            print(f"   ✅ {arch} LSTM - Validation Loss: {val_loss:.4f}")
            
            self.models[f"lstm_{arch}"] = model
        
        # Train ML models
        print("🌳 Training ML models...")
        self.create_ml_models()
        
        for name, model in self.models.items():
            if name.startswith('lstm_'):
                continue
                
            print(f"📊 Training {name}...")
            
            # Scale features for ML models
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_ml)
            X_val_scaled = scaler.transform(X_val_ml)
            
            # Train model
            model.fit(X_train_scaled, y_ml)
            
            # Evaluate
            val_pred = model.predict(X_val_scaled)
            val_mse = np.mean((val_pred - y_val_ml) ** 2)
            print(f"   ✅ {name} - Validation MSE: {val_mse:.4f}")
            
            # Store scaler
            self.scalers[name] = scaler
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        print("✅ Ensemble training completed!")
    
    def predict_ensemble(self, df: pd.DataFrame, confidence_threshold: float = 0.1) -> Dict:
        """
        Make predictions using all models and combine them.
        """
        print("🔮 Making ensemble predictions...")
        
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
                    
                    # Calculate confidence based on prediction variance
                    if len(pred) > 1:
                        confidence = 1 / (1 + np.std(pred[-10:]))  # Last 10 predictions
                        confidences[name] = confidence
                    else:
                        confidences[name] = 0.5
        
        # ML predictions
        X_ml = features[self.sequence_length:]
        
        for name, model in self.models.items():
            if not name.startswith('lstm_'):
                if name in self.scalers:
                    X_scaled = self.scalers[name].transform(X_ml)
                    pred = model.predict(X_scaled)
                    predictions[name] = pred
                    
                    # For ML models, use model's inherent confidence if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_scaled)
                        confidence = np.max(proba, axis=1).mean()
                    else:
                        confidence = 0.7  # Default confidence for ML models
                    confidences[name] = confidence
        
        # Combine predictions with weighted average
        if predictions:
            # Calculate weights based on confidence
            total_confidence = sum(confidences.values())
            weights = {name: conf / total_confidence for name, conf in confidences.items()}
            
            # Weighted ensemble prediction
            ensemble_pred = np.zeros(len(list(predictions.values())[0]))
            for name, pred in predictions.items():
                ensemble_pred += weights[name] * pred
            
            # Calculate ensemble confidence
            ensemble_confidence = np.mean(list(confidences.values()))
            
            # Filter predictions based on confidence threshold
            if ensemble_confidence < confidence_threshold:
                print(f"⚠️ Low confidence prediction: {ensemble_confidence:.3f}")
            
            return {
                'ensemble_prediction': ensemble_pred,
                'ensemble_confidence': ensemble_confidence,
                'individual_predictions': predictions,
                'model_weights': weights,
                'model_confidences': confidences
            }
        else:
            return {}
    
    def save_ensemble(self, save_dir: str = 'models'):
        """
        Save all models and scalers.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            if name.startswith('lstm_'):
                model.save(f'{save_dir}/{name}_ensemble.h5')
            else:
                joblib.dump(model, f'{save_dir}/{name}_ensemble.pkl')
        
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{save_dir}/{name}_scaler_ensemble.pkl')
        
        # Save feature importance
        if self.feature_importance:
            joblib.dump(self.feature_importance, f'{save_dir}/feature_importance_ensemble.pkl')
        
        print(f"✅ Ensemble models saved to {save_dir}")
    
    def load_ensemble(self, save_dir: str = 'models'):
        """
        Load all models and scalers.
        """
        # Load LSTM models
        lstm_architectures = ['standard', 'bidirectional', 'cnn_lstm', 'attention_lstm', 'deep_lstm']
        
        for arch in lstm_architectures:
            model_path = f'{save_dir}/lstm_{arch}_ensemble.h5'
            if os.path.exists(model_path):
                self.models[f'lstm_{arch}'] = tf.keras.models.load_model(model_path)
        
        # Load ML models
        ml_models = ['random_forest', 'gradient_boosting']
        for name in ml_models:
            model_path = f'{save_dir}/{name}_ensemble.pkl'
            scaler_path = f'{save_dir}/{name}_scaler_ensemble.pkl'
            
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
            if os.path.exists(scaler_path):
                self.scalers[name] = joblib.load(scaler_path)
        
        # Load feature importance
        importance_path = f'{save_dir}/feature_importance_ensemble.pkl'
        if os.path.exists(importance_path):
            self.feature_importance = joblib.load(importance_path)
        
        print(f"✅ Ensemble models loaded from {save_dir}")

def test_ensemble_model():
    """Test the ensemble model."""
    print("🧪 Testing ensemble model...")
    
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
    ensemble = EnsembleModel(sequence_length=30, n_features=n_features)
    ensemble.train_ensemble(df)
    
    # Make predictions
    predictions = ensemble.predict_ensemble(df)
    
    if predictions:
        print(f"📊 Ensemble prediction shape: {predictions['ensemble_prediction'].shape}")
        print(f"🎯 Ensemble confidence: {predictions['ensemble_confidence']:.3f}")
        print(f"⚖️ Model weights: {predictions['model_weights']}")
    
    # Save ensemble
    ensemble.save_ensemble()
    
    return ensemble

if __name__ == "__main__":
    test_ensemble_model() 