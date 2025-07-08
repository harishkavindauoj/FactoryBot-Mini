"""Train the student model using soft labels and advanced techniques."""

import os
import numpy as np
import tensorflow as tf
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError, Huber
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import joblib

from src.model import build_model


def train_student_model(config, force_retrain=False):
    best_model_path = "models/best_student_model.h5"
    final_model_path = "models/final_student_model.h5"

    # ✅ Skip training if both models already exist
    if not force_retrain and os.path.exists(best_model_path) and os.path.exists(final_model_path):

        if force_retrain:
            print("[Trainer] 🔁 Force retrain enabled. Ignoring existing model files...")


        print("[Trainer] ✅ Trained models found. Skipping training and loading final model...")
        model = tf.keras.models.load_model(final_model_path)
        history = None  # No history if training is skipped
        return model, history

    print("[Trainer] 🔍 Loading features and soft labels...")
    X = np.load(config['data']['features_cache'])
    y_risk = np.load("data/processed/risk_labels.npy")
    y_ttf = np.load("data/processed/ttf_labels.npy")

    print("[Trainer] ⚙️ Preparing optimizer and loss functions...")
    optimizer = Adam(
        learning_rate=config['training']['initial_lr'],
        clipnorm=config['advanced'].get('gradient_clip_norm', 1.0)
    )

    label_smoothing = config['advanced'].get('label_smoothing', 0.0)
    print(f"[Trainer] 🎛️ Label smoothing = {label_smoothing}")
    loss_risk = CategoricalCrossentropy(label_smoothing=label_smoothing)

    loss_ttf = Huber()

    loss_weights = config['model']['loss_weights']
    losses = {
        "risk_level": loss_risk,
        "ttf": loss_ttf
    }

    class_weights = None
    sample_weight = None
    if config['advanced'].get('use_class_weights', False):
        print("[Trainer] 🧮 Computing class weights for risk classification...")
        y_classes = np.argmax(y_risk, axis=1)
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_classes), y=y_classes)
        class_weights = {i: w for i, w in enumerate(weights)}
        print(f"[Trainer] 📊 Class weights: {class_weights}")

        print("[Trainer] 🏋️ Applying class weights via sample_weight...")
        risk_sample_weights = np.array([class_weights[c] for c in y_classes])
        sample_weight = {
            "risk_level": risk_sample_weights,
            "ttf": np.ones(len(y_ttf))
        }

    print("[Trainer] 🏗️ Building model architecture...")
    model = build_model(config)

    print("[Trainer] 🧪 Compiling model...")
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            "risk_level": "accuracy",
            "ttf": "mae"
        }
    )

    print("[Trainer] 🎯 Preparing callbacks...")
    callbacks = [
        EarlyStopping(patience=config['training']['early_stopping_patience'], restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(config['training']['lr_factor']),
            patience=int(config['training']['lr_patience']),
            min_lr=float(config['training']['min_lr']),
            verbose=1
        ),
        ModelCheckpoint(best_model_path, save_best_only=True, monitor="val_loss", verbose=1)
    ]

    print("[Trainer] 🚀 Starting model training...")
    history = model.fit(
        X,
        {"risk_level": y_risk, "ttf": y_ttf},
        validation_split=config['training']['validation_split'],
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        sample_weight=sample_weight,
        callbacks=callbacks
    )

    print("[Trainer] 💾 Saving final trained model...")
    os.makedirs("models", exist_ok=True)
    model.save(final_model_path)
    print("[Trainer] ✅ Student model training complete and saved.")

    return model, history


