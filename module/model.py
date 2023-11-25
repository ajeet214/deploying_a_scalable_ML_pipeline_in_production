"""
model.py

This module contains functions related to training a machine learning model, making predictions,
and computing evaluation metrics.

Author:
Ajeet Kumar Verma
"""
import sys
import pandas as pd
import hydra
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score

# Import local module for data processing
try:
    from module.data import process_data
except ModuleNotFoundError:
    sys.path.append('./')
    from module.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    logistic_reg_model = LogisticRegression(max_iter=1000, random_state=8071)
    logistic_reg_model.fit(X_train, y_train.ravel())
    return logistic_reg_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Args:
        y (np.array): Known labels, binarized.
        preds (np.array): Predicted labels, binarized.
    Returns:
        precision : float
        recall : float
        fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_metrics_with_slices_data(
        df, cat_columns, label,
        encoder, lb, model, slice_output_path):
    """
    Compute evaluation metrics for each category within categorical features.

    Parameters:
    - df: pd.DataFrame
      Input DataFrame.
    - cat_columns: list
      List of categorical features.
    - label: str
      Label column name.
    - encoder: sklearn.preprocessing._encoders.OneHotEncoder
      Trained OneHotEncoder.
    - lb: sklearn.preprocessing._label.LabelBinarizer
      Trained LabelBinarizer.
    - model: Trained machine learning model.
    - slice_output_path: str
      Output path for the metrics.

    Returns:
    - metrics: pd.DataFrame
      DataFrame containing evaluation metrics for each category.
    """

    rows_list = list()
    for feature in cat_columns:
        for category in df[feature].unique():
            row = {}
            tmp_df = df[df[feature] == category]

            x, y, _, _ = process_data(
                X=tmp_df,
                categorical_features=cat_columns,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb
            )

            preds = inference(model, x)
            precision, recall, fbeta = compute_model_metrics(y, preds)

            row['feature'] = feature
            row['precision'] = precision
            row['recall'] = recall
            row['f1'] = fbeta
            row['category'] = category
            rows_list.append(row)

    metrics = pd.DataFrame(
        rows_list,
        columns=[
            "feature",
            "precision",
            "recall",
            "f1",
            "category"])
    metrics.to_csv(slice_output_path, index=False)
    return metrics