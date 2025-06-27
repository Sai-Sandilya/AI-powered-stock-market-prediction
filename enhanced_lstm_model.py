"""
Enhanced LSTM Model with Attention Mechanism
Advanced neural network architectures for better stock prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Attention, Concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AttentionLSTM:
    """LSTM with attention mechanism for sequence modeling."""
    
    def __init__(self, sequence_length: int, n_features: int, n_outputs: int = 1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_attention_lstm(self, lstm_units: int = 128, dropout_rate: float = 0.3) -> keras.Model:
        """
        Build LSTM model with attention mechanism.
        """
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # First LSTM layer
        lstm1 = LSTM(lstm_units, return_sequences=True, activation='tanh')(inputs)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(dropout_rate)(lstm1)
        
        # Second LSTM layer with attention
        lstm2 = LSTM(lstm_units, return_sequences=True, activation='tanh')(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Dropout(dropout_rate)(lstm2)
        
        # Attention mechanism
        attention = layers.MultiHeadAttention(
            num_heads=8, 
            key_dim=lstm_units,
            dropout=dropout_rate
        )(lstm2, lstm2)
        
        # Residual connection
        attention_output = layers.Add()([lstm2, attention])
        attention_output = layers.LayerNormalization()(attention_output)
        
        # Global average pooling with attention weights
        attention_weights = Dense(1, activation='tanh')(attention_output)
        attention_weights = Dense(1, activation='softmax')(attention_weights)
        weighted_output = layers.Multiply()([attention_output, attention_weights])
        pooled_output = layers.GlobalAveragePooling1D()(weighted_output)
        
        # Dense layers
        dense1 = Dense(lstm_units // 2, activation='relu')(pooled_output)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(dropout_rate)(dense1)
        
        dense2 = Dense(lstm_units // 4, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(dropout_rate)(dense2)
        
        # Output layer
        outputs = Dense(self.n_outputs, activation='linear')(dense2)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

class BidirectionalLSTM:
    """Bidirectional LSTM for capturing both forward and backward dependencies."""
    
    def __init__(self, sequence_length: int, n_features: int, n_outputs: int = 1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_bidirectional_lstm(self, lstm_units: int = 128, dropout_rate: float = 0.3) -> keras.Model:
        """
        Build bidirectional LSTM model.
        """
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Bidirectional LSTM layers
        bi_lstm1 = layers.Bidirectional(
            LSTM(lstm_units, return_sequences=True, activation='tanh')
        )(inputs)
        bi_lstm1 = BatchNormalization()(bi_lstm1)
        bi_lstm1 = Dropout(dropout_rate)(bi_lstm1)
        
        bi_lstm2 = layers.Bidirectional(
            LSTM(lstm_units // 2, return_sequences=True, activation='tanh')
        )(bi_lstm1)
        bi_lstm2 = BatchNormalization()(bi_lstm2)
        bi_lstm2 = Dropout(dropout_rate)(bi_lstm2)
        
        bi_lstm3 = layers.Bidirectional(
            LSTM(lstm_units // 4, return_sequences=False, activation='tanh')
        )(bi_lstm2)
        bi_lstm3 = BatchNormalization()(bi_lstm3)
        bi_lstm3 = Dropout(dropout_rate)(bi_lstm3)
        
        # Dense layers
        dense1 = Dense(lstm_units, activation='relu')(bi_lstm3)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(dropout_rate)(dense1)
        
        dense2 = Dense(lstm_units // 2, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(dropout_rate)(dense2)
        
        # Output layer
        outputs = Dense(self.n_outputs, activation='linear')(dense2)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

class StackedLSTM:
    """Deep stacked LSTM with residual connections."""
    
    def __init__(self, sequence_length: int, n_features: int, n_outputs: int = 1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_stacked_lstm(self, lstm_units: int = 128, dropout_rate: float = 0.3) -> keras.Model:
        """
        Build deep stacked LSTM model with residual connections.
        """
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Initial LSTM layer
        x = LSTM(lstm_units, return_sequences=True, activation='tanh')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Stacked LSTM layers with residual connections
        for i in range(3):
            residual = x
            
            # LSTM block
            x = LSTM(lstm_units, return_sequences=True, activation='tanh')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
            
            # Residual connection
            if x.shape[-1] == residual.shape[-1]:
                x = layers.Add()([x, residual])
            
            # Reduce units for deeper layers
            if i == 1:
                lstm_units = lstm_units // 2
                x = Dense(lstm_units)(x)
        
        # Final LSTM layer
        x = LSTM(lstm_units, return_sequences=False, activation='tanh')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Dense layers
        dense1 = Dense(lstm_units * 2, activation='relu')(x)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(dropout_rate)(dense1)
        
        dense2 = Dense(lstm_units, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(dropout_rate)(dense2)
        
        # Output layer
        outputs = Dense(self.n_outputs, activation='linear')(dense2)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

class TransformerLSTM:
    """Hybrid Transformer-LSTM model for sequence modeling."""
    
    def __init__(self, sequence_length: int, n_features: int, n_outputs: int = 1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_transformer_lstm(self, lstm_units: int = 128, dropout_rate: float = 0.3) -> keras.Model:
        """
        Build hybrid Transformer-LSTM model.
        """
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Feature embedding
        embedding_dim = lstm_units
        x = Dense(embedding_dim)(inputs)
        x = layers.LayerNormalization()(x)
        
        # Transformer blocks
        for _ in range(2):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=8, 
                key_dim=embedding_dim // 8,
                dropout=dropout_rate
            )(x, x)
            
            # Add & Norm
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization()(x)
            
            # Feed-forward network
            ffn = Dense(embedding_dim * 4, activation='relu')(x)
            ffn = Dropout(dropout_rate)(ffn)
            ffn = Dense(embedding_dim)(ffn)
            
            # Add & Norm
            x = layers.Add()([x, ffn])
            x = layers.LayerNormalization()(x)
        
        # LSTM layer for temporal modeling
        x = LSTM(lstm_units, return_sequences=True, activation='tanh')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = LSTM(lstm_units // 2, return_sequences=False, activation='tanh')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Dense layers
        dense1 = Dense(lstm_units, activation='relu')(x)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(dropout_rate)(dense1)
        
        dense2 = Dense(lstm_units // 2, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(dropout_rate)(dense2)
        
        # Output layer
        outputs = Dense(self.n_outputs, activation='linear')(dense2)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

class EnsembleLSTM:
    """Ensemble of multiple LSTM architectures."""
    
    def __init__(self, sequence_length: int, n_features: int, n_outputs: int = 1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.models = {}
        self.scaler = MinMaxScaler()
        
    def build_ensemble(self, lstm_units: int = 128, dropout_rate: float = 0.3):
        """
        Build ensemble of different LSTM architectures.
        """
        # Attention LSTM
        attention_lstm = AttentionLSTM(self.sequence_length, self.n_features, self.n_outputs)
        self.models['attention_lstm'] = attention_lstm.build_attention_lstm(lstm_units, dropout_rate)
        
        # Bidirectional LSTM
        bi_lstm = BidirectionalLSTM(self.sequence_length, self.n_features, self.n_outputs)
        self.models['bidirectional_lstm'] = bi_lstm.build_bidirectional_lstm(lstm_units, dropout_rate)
        
        # Stacked LSTM
        stacked_lstm = StackedLSTM(self.sequence_length, self.n_features, self.n_outputs)
        self.models['stacked_lstm'] = stacked_lstm.build_stacked_lstm(lstm_units, dropout_rate)
        
        # Transformer LSTM
        transformer_lstm = TransformerLSTM(self.sequence_length, self.n_features, self.n_outputs)
        self.models['transformer_lstm'] = transformer_lstm.build_transformer_lstm(lstm_units, dropout_rate)
        
        return self.models
    
    def compile_models(self, learning_rate: float = 0.001):
        """
        Compile all models in the ensemble.
        """
        for name, model in self.models.items():
            model.compile(
                optimizer=optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae']
            )
            print(f"✅ Compiled {name}")
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 32):
        """
        Train all models in the ensemble.
        """
        training_history = {}
        
        for name, model in self.models.items():
            print(f"\n🚀 Training {name}...")
            
            # Callbacks
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            training_history[name] = history.history
            
            # Evaluate model
            val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
            print(f"📊 {name} - Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        return training_history
    
    def predict_ensemble(self, X):
        """
        Make predictions using all models and return ensemble prediction.
        """
        predictions = {}
        
        for name, model in self.models.items():
            pred = model.predict(X, verbose=0)
            predictions[name] = pred.flatten()
        
        # Ensemble prediction (simple average)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        return ensemble_pred, predictions
    
    def save_ensemble(self, base_path: str = "models"):
        """
        Save all models in the ensemble.
        """
        import os
        os.makedirs(base_path, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(base_path, f"{name}.h5")
            model.save(model_path)
            print(f"💾 Saved {name} to {model_path}")
    
    def load_ensemble(self, base_path: str = "models"):
        """
        Load all models in the ensemble.
        """
        import os
        
        for name in self.models.keys():
            model_path = os.path.join(base_path, f"{name}.h5")
            if os.path.exists(model_path):
                self.models[name] = keras.models.load_model(model_path)
                print(f"📂 Loaded {name} from {model_path}")

def create_enhanced_lstm_model(model_type: str = 'ensemble', **kwargs) -> EnsembleLSTM:
    """
    Create enhanced LSTM model based on specified type.
    
    Args:
        model_type: 'attention', 'bidirectional', 'stacked', 'transformer', or 'ensemble'
        **kwargs: Model parameters
    """
    sequence_length = kwargs.get('sequence_length', 60)
    n_features = kwargs.get('n_features', 50)
    n_outputs = kwargs.get('n_outputs', 1)
    
    if model_type == 'ensemble':
        model = EnsembleLSTM(sequence_length, n_features, n_outputs)
        model.build_ensemble(**kwargs)
        return model
    elif model_type == 'attention':
        model = AttentionLSTM(sequence_length, n_features, n_outputs)
        return model.build_attention_lstm(**kwargs)
    elif model_type == 'bidirectional':
        model = BidirectionalLSTM(sequence_length, n_features, n_outputs)
        return model.build_bidirectional_lstm(**kwargs)
    elif model_type == 'stacked':
        model = StackedLSTM(sequence_length, n_features, n_outputs)
        return model.build_stacked_lstm(**kwargs)
    elif model_type == 'transformer':
        model = TransformerLSTM(sequence_length, n_features, n_outputs)
        return model.build_transformer_lstm(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def prepare_data_for_lstm(df: pd.DataFrame, target_col: str = 'Close', 
                         sequence_length: int = 60, test_size: float = 0.2) -> tuple:
    """
    Prepare data for LSTM training.
    """
    # Select features (exclude Date and target column)
    feature_cols = [col for col in df.columns if col not in ['Date', target_col]]
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(df[target_col].iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    return X_train, X_test, y_train, y_test, scaler, feature_cols 