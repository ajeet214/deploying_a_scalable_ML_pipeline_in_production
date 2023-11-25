"""
train_model.py

This script trains a machine learning model and saves it. It uses data processing and model-related functions
from the 'module.data' and 'module.model' modules.

Usage:
- Run this script to train a machine learning model.
- The script reads the configuration from 'config.yml' file.

Author:
Ajeet Kumar Verma
"""
import os
import sys
import pickle
import logging
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

# Import necessary functions from local modules
try:
    from module.data import process_data
    from module.model import (
        inference,
        compute_model_metrics,
        train_model,
        compute_metrics_with_slices_data
    )
except ModuleNotFoundError:
    sys.path.append('./')
    from module.data import process_data
    from module.model import (
        inference,
        compute_model_metrics,
        train_model,
        compute_metrics_with_slices_data
    )

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def training(config: DictConfig):
    """
    Trains a machine learning model and saves it.

    Parameters:
    - config: DictConfig
      Configuration dictionary.

    Returns:
    - None
    """

    # Extract configuration parameters
    CATEGORY_FEATURES = config['main']['cat_features']
    LABEL = config['main']['label']
    TEST_SIZE = config['main']['test_size']
    SLICE_OUTPUT_PATH = config['main']['slice_output_path']
    MODEL_PATH = config['main']['model_path']
    DATA_PATH = config['main']['data_path']

    # Log configuration information
    logger.info(f"Hydra config: {config}")

    # Read training data
    logger.info("Read training data...")
    df = pd.read_csv(DATA_PATH)
    logger.info(df.describe())

    # Split data into train and test sets
    logger.info("Split data into train and test sets...")
    train, test = train_test_split(df, test_size=TEST_SIZE)

    # Process data for training and inference
    logger.info("Process data for training and inference...")
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CATEGORY_FEATURES, label=LABEL, training=True)

    X_test, y_test, encoder, lb = process_data(
        X=test,
        categorical_features=CATEGORY_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Train machine learning model
    logger.info("Train model...")
    model = train_model(X_train, y_train)
    logger.info(model)

    # Save the trained model
    logger.info("Save the model...")
    if not os.path.exists("model/"):
        os.mkdir("model/")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump([encoder, lb, model], f)
    logger.info("Model saved.")

    # Inference with the trained model
    logger.info("Inference with the trained model...")
    preds = inference(model, X_test)

    # Calculate model metrics
    logger.info("Calculate model metrics...")
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    logger.info(f">>>Precision: {precision}")
    logger.info(f">>>Recall: {recall}")
    logger.info(f">>>Fbeta: {fbeta}")

    # Calculate model metrics on slices data
    logger.info("Calculate model metrics on slices data...")
    metrics = compute_metrics_with_slices_data(
        df=test,
        cat_columns=CATEGORY_FEATURES,
        label=LABEL,
        encoder=encoder,
        lb=lb,
        model=model,
        slice_output_path=SLICE_OUTPUT_PATH
    )
    logger.info(f">>>Metrics with slices data: {metrics}")
