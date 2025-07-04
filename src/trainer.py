"""Handles training of the student model with improved TTF handling."""
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau

from src.model import build_model
import numpy as np

from sklearn.preprocessing import StandardScaler
import pickle
import os


def train_student_model(config):
    print("[Trainer] Training student model...")

    # Load data
    X = np.load(config['data']['features_cache'])
    y_risk = np.load("data/processed/risk_labels.npy")
    y_ttf = np.load("data/processed/ttf_labels.npy")

    # Use log transformation instead of standardization for TTF
    # This preserves the relationship better for large values
    print("[Trainer] Applying log transformation to TTF values...")
    y_ttf_log = np.log1p(y_ttf)  # log1p = log(1 + x) to handle zeros

    # Store transformation parameters
    os.makedirs("models", exist_ok=True)
    transformation_params = {
        'method': 'log1p',
        'original_mean': np.mean(y_ttf),
        'original_std': np.std(y_ttf),
        'log_mean': np.mean(y_ttf_log),
        'log_std': np.std(y_ttf_log)
    }

    with open("models/ttf_transform.pkl", "wb") as f:
        pickle.dump(transformation_params, f)

    print(f"[Trainer] TTF statistics - Original: mean={np.mean(y_ttf):.1f}, std={np.std(y_ttf):.1f}")
    print(f"[Trainer] TTF statistics - Log transformed: mean={np.mean(y_ttf_log):.3f}, std={np.std(y_ttf_log):.3f}")

    # Build model
    model = build_model(config)

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config['training'].get('early_stopping_patience', 5),
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train model with log-transformed TTF values
    print("[Trainer] Starting training with log-transformed TTF...")
    history = model.fit(
        X, {"risk_level": y_risk, "ttf": y_ttf_log},
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        validation_split=config['training']['validation_split'],
        callbacks=callbacks,
        verbose=1
    )

    # Save model in modern format
    model.save("models/student_model.keras")
    print("[Trainer] Model saved in Keras format")

    # Evaluate on original scale
    print("[Trainer] Evaluating on original TTF scale...")
    val_split = config['training']['validation_split']
    split_idx = int(len(X) * (1 - val_split))
    X_val = X[split_idx:]
    y_ttf_val = y_ttf[split_idx:]

    # Get predictions and convert back to original scale
    predictions = model.predict(X_val, verbose=0)
    ttf_pred_log = predictions[1] if isinstance(predictions, list) else predictions['ttf']
    ttf_pred_original = np.expm1(ttf_pred_log.flatten())  # expm1 = exp(x) - 1

    # Calculate MAE on original scale
    mae_original = np.mean(np.abs(ttf_pred_original - y_ttf_val))
    print(f"[Trainer] Final TTF MAE on original scale: {mae_original:.1f} hours ({mae_original / 24:.1f} days)")

    return model, history


def load_ttf_transformer():
    """Load the TTF transformation parameters."""
    with open("models/ttf_transform.pkl", "rb") as f:
        return pickle.load(f)


def transform_ttf_predictions(predictions_log):
    """Transform log predictions back to original scale."""
    return np.expm1(predictions_log)


def evaluate_model(model, config):
    """Evaluate model with proper TTF transformation."""
    print("[Evaluate] Evaluating student model...")

    # Load test data
    X_test = np.load(config['data']['features_cache'])
    y_ttf_test = np.load("data/processed/ttf_labels.npy")

    # Get predictions
    predictions = model.predict(X_test, verbose=1)

    # Transform back to original scale
    ttf_pred_log = predictions[1] if isinstance(predictions, list) else predictions['ttf']
    ttf_pred_original = transform_ttf_predictions(ttf_pred_log.flatten())

    # Calculate MAE
    mae_hours = np.mean(np.abs(ttf_pred_original - y_ttf_test))
    print(f"Mean Absolute Error (TTF): {mae_hours:.2f} hours ({mae_hours / 24:.1f} days)")

    return mae_hours


# Alternative: Simpler approach without transformation
def train_student_model_simple(config):
    """Simplified training with just callbacks, no TTF transformation."""
    print("[Trainer] Training student model (simple approach)...")

    # Load data
    X = np.load(config['data']['features_cache'])
    y_risk = np.load("data/processed/risk_labels.npy")
    y_ttf = np.load("data/processed/ttf_labels.npy")

    # Build model
    model = build_model(config)

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config['training'].get('early_stopping_patience', 5),
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train model with original TTF values
    print("[Trainer] Starting training with original TTF values...")
    history = model.fit(
        X, {"risk_level": y_risk, "ttf": y_ttf},
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        validation_split=config['training']['validation_split'],
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    model.save("models/student_model.keras")
    print("[Trainer] Model saved in Keras format")

    return model, history