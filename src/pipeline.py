"""End-to-end pipeline: load data, distill, train, evaluate."""
from src.preprocessor import preprocess_and_save
from src.trainer import train_student_model
from src.utils import load_config
from src.distiller import generate_soft_labels
from src.evaluate import evaluate_model
import os


def run_pipeline():
    config = load_config("config/config.yaml")

    # Step 0: Preprocess raw data into feature matrix
    preprocess_and_save(config)

    # Step 1: Generate soft labels using Gemini
    # Emergency fast mode (10 minutes)
    config['use_sampling'] = True
    config['sample_ratio'] = 0.1
    config['force_regenerate'] = True  # optional: force regeneration
    generate_soft_labels(config)

    # Step 2: Train student model
    model, history = train_student_model(config, force_retrain=True)

    # Step 3: Evaluate
    evaluate_model(model, config)