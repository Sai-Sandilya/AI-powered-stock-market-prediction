import pandas as pd
import numpy as np
import os
import joblib
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- Configuration ---
FEATURE_DATA_PATH = 'data/featured_data'
MODEL_PATH = 'models'
SYMBOL_TO_TRAIN = 'AAPL'
TARGET_COLUMN = 'Close'
N_TRIALS = 25
EPOCHS = 50
PATIENCE = 10

os.makedirs(MODEL_PATH, exist_ok=True)

def create_sequences(features, target, sequence_length):
    xs, ys = [], []
    for i in range(len(features) - sequence_length):
        xs.append(features[i:i + sequence_length])
        ys.append(target[i + sequence_length])
    return np.array(xs), np.array(ys)

def objective(trial, X_train, y_train, X_val, y_val, feature_count):
    sequence_length = trial.suggest_int('sequence_length', 30, 120)
    lstm_units = trial.suggest_categorical('lstm_units', [64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)

    # Pruning trial if there isn't enough data for a single batch
    if len(X_train_seq) < batch_size or len(X_val_seq) < batch_size:
        raise optuna.exceptions.TrialPruned()

    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(sequence_length, feature_count)),
        Dropout(dropout_rate),
        LSTM(units=lstm_units // 2),
        Dropout(dropout_rate),
        Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss_fn)

    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=EPOCHS,
        batch_size=batch_size,
        validation_data=(X_val_seq, y_val_seq),
        callbacks=[early_stopping, lr_scheduler],
        verbose=0 # Quieter output for Optuna
    )

    return min(history.history['val_loss'])

def train_and_evaluate_model(symbol, df, feature_columns, target_column, model_path_prefix):
    features = df[feature_columns].values
    target = df[target_column].values.reshape(-1, 1)

    scaler_features = MinMaxScaler()
    features_scaled = scaler_features.fit_transform(features)
    scaler_target = MinMaxScaler()
    target_scaled = scaler_target.fit_transform(target)
    
    train_size = int(len(df) * 0.8)
    X_train, X_val = features_scaled[:train_size], features_scaled[train_size:]
    y_train, y_val = target_scaled[:train_size], target_scaled[train_size:]
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, len(feature_columns)), n_trials=N_TRIALS)
    
    best_params = study.best_params
    
    X_train_final, y_train_final = create_sequences(X_train, y_train, best_params['sequence_length'])
    X_val_final, y_val_final = create_sequences(X_val, y_val, best_params['sequence_length'])
    
    final_model = Sequential([
        LSTM(units=best_params['lstm_units'], return_sequences=True, input_shape=(best_params['sequence_length'], len(feature_columns))),
        Dropout(best_params['dropout_rate']),
        LSTM(units=best_params['lstm_units'] // 2),
        Dropout(best_params['dropout_rate']),
        Dense(1)
    ])
    
    final_optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    final_loss = tf.keras.losses.MeanSquaredError()
    final_model.compile(optimizer=final_optimizer, loss=final_loss)
    
    final_model.fit(
        X_train_final, y_train_final,
        epochs=EPOCHS,
        batch_size=best_params['batch_size'],
        validation_data=(X_val_final, y_val_final),
        callbacks=[EarlyStopping(monitor='val_loss', patience=PATIENCE)],
        verbose=1
    )

    model_save_path = f"{model_path_prefix}_model.keras"
    scaler_features_path = f"{model_path_prefix}_scaler_features.pkl"
    scaler_target_path = f"{model_path_prefix}_scaler_target.pkl"
    metadata_path = f"{model_path_prefix}_metadata.pkl"

    final_model.save(model_save_path)
    joblib.dump(scaler_features, scaler_features_path)
    joblib.dump(scaler_target, scaler_target_path)
    
    metadata = {
        'sequence_length': best_params['sequence_length'],
        'feature_columns': feature_columns
    }
    joblib.dump(metadata, metadata_path)
    
    return model_save_path, scaler_features_path, scaler_target_path, metadata_path

if __name__ == '__main__':
    input_file = os.path.join(FEATURE_DATA_PATH, f"{SYMBOL_TO_TRAIN}_featured_data.parquet")
    df = pd.read_parquet(input_file)
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]
    model_prefix = os.path.join(MODEL_PATH, f"{SYMBOL_TO_TRAIN}_institutional")
    train_and_evaluate_model(SYMBOL_TO_TRAIN, df, feature_cols, TARGET_COLUMN, model_prefix)
    print("Institutional model training script finished.")