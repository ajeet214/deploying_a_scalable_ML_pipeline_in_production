"""
Tests for the data processing and model functionality.

This module contains a set of pytest tests to validate the correctness of data processing,
model training, and model inference in the machine learning pipeline.

Author: Ajeet Kumar Verma
"""
import pickle
import pandas as pd
import pytest
import pandas.api.types as pdtypes
from sklearn.model_selection import train_test_split
from module.data import process_data
from module.model import inference, compute_model_metrics

# Fake categorical features for testing
fake_categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",

]


@pytest.fixture(scope="module")
def data():
    """
       Fixture to load the test dataset.
       """
    return pd.read_csv("data/census_clean.csv", skipinitialspace=True)


def test_column_presence_and_type(data):
    """
        Test to check the presence and data types of required columns.

        Args:
            data (pd.DataFrame): Test dataset.
    """

    required_columns = {
        "age": pdtypes.is_int64_dtype,
        "workclass": pdtypes.is_object_dtype,
        "fnlgt": pdtypes.is_int64_dtype,
        "education": pdtypes.is_object_dtype,
        "education-num": pdtypes.is_int64_dtype,
        "marital-status": pdtypes.is_object_dtype,
        "occupation": pdtypes.is_object_dtype,
        "relationship": pdtypes.is_object_dtype,
        "race": pdtypes.is_object_dtype,
        "sex": pdtypes.is_object_dtype,
        "capital-gain": pdtypes.is_int64_dtype,
        "capital-loss": pdtypes.is_int64_dtype,
        "hours-per-week": pdtypes.is_int64_dtype,
        "native-country": pdtypes.is_object_dtype,
        "salary": pdtypes.is_object_dtype,
    }

    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    # Check that the columns are of the right dtype
    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(
            data[col_name]
        ), f"Column {col_name} failed test {format_verification_funct}"


def workclass_values(data):
    """
    Test for the unique values of the 'workclass' column.

    Args:
        data (pd.DataFrame): Test dataset.
    """
    expected_values = {
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    }

    assert set(data["workclass"].unique()) == expected_values


def education_values(data):
    """
    Test the values in the 'education' column of the dataset.

    Args:
        data (pd.DataFrame): Dataset to check for valid values in the 'education' column.
    """
    # Define the expected values for the 'education' column
    expected_values = {
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    }

    # Assert that the unique values in the 'education' column match the expected values
    assert set(data["education"].unique()) == expected_values


def marital_status_values(data):
    """
    Test the values in the 'marital-status' column of the dataset.

    Args:
        data (pd.DataFrame): Dataset to check for valid values in the 'marital-status' column.
    """
    # Define the expected values for the 'marital-status' column
    expected_values = {
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    }

    # Assert that the unique values in the 'marital-status' column match the expected values
    assert set(data["marital-status"].unique()) == expected_values


def occupation_values(data):
    """
    Test the values in the 'occupation' column of the dataset.

    Args:
        data (pd.DataFrame): Dataset to check for valid values in the 'occupation' column.
    """
    # Define the expected values for the 'occupation' column
    expected_values = {
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    }

    # Assert that the unique values in the 'occupation' column match the expected values
    assert set(data["occupation"].unique()) == expected_values


def relationship_values(data):
    """
    Test the values in the 'relationship' column of the dataset.

    Args:
        data (pd.DataFrame): Dataset to check for valid values in the 'relationship' column.
    """
    # Define the expected values for the 'relationship' column
    expected_values = {
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    }

    # Assert that the unique values in the 'relationship' column match the expected values
    assert set(data["relationship"].unique()) == expected_values


def test_sex_values(data):
    """
    Test the values in the 'sex' column of the dataset.

    Args:
        data (pd.DataFrame): Dataset to check for valid values in the 'sex' column.
    """
    # Define the expected values for the 'sex' column
    expected_values = {"Male", "Female"}

    # Assert that the unique values in the 'sex' column match the expected values
    assert set(data["sex"]) == expected_values


def test_salary_values(data):
    """
    Test to check the unique values of the 'salary' column.

    Args:
        data (pd.DataFrame): Test dataset.
    """
    # Define expected unique values for 'salary'
    expected_values = {
        "<=50K",
        ">50K"
    }

    # Assert that the unique values of the 'salary' column match the expected values
    assert set(data["salary"]) == expected_values


def test_column_ranges(data):
    """
    Test to check if numerical columns fall within specified ranges.

    Args:
        data (pd.DataFrame): Test dataset.
    """
    ranges = {
        "age": (17, 90),
        "education-num": (1, 16),
        "hours-per-week": (1, 99),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
    }

    for col_name, (minimum, maximum) in ranges.items():
        assert data[col_name].min() >= minimum
        assert data[col_name].max() <= maximum


def test_column_values(data):
    """
    Test to check if there are no null values in any column.

    Args:
        data (pd.DataFrame): Test dataset.
    """
    # Check that the columns are of the right dtype
    for col_name in data.columns.values:
        assert not data[col_name].isnull().any(
        ), f"Column {col_name} has null values"


def test_model_input(data):
    """
    Test to check that features used as model input do not have null values.

    Args:
        data (pd.DataFrame): Test dataset.
    """
    # Iterate over each column in the dataset
    for col_name in data.columns.values:
        # Assert that the feature does not have any null values
        assert not data[col_name].isnull().any(
        ), f"Features {col_name} has null values"


def test_inference(data):
    """
    Test the inference process for the trained model.

    Args:
        data (pd.DataFrame): Test dataset.

    Notes:
        Assumes the existence of a trained logistic regression model and associated encoders.
    """
    # Split the data into training and test sets
    _, test_df = train_test_split(data, test_size=0.20)

    # Load the pre-trained encoder, label binarizer, and logistic regression model
    [encoder, lb, lr_model] = pickle.load(open("model/lr_model.pkl", "rb"))

    # Process the test data for model inference
    X_test, y_test, _, _ = process_data(
        X=test_df,
        categorical_features=fake_categorical_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Make predictions using the logistic regression model
    preds = inference(lr_model, X_test)

    # Assert that the length of predictions matches the length of the test set
    assert len(preds) == len(X_test)


def test_output_metrics(data):
    """
    Test for the computation of precision, recall, and F-beta score.

    Args:
        data (pd.DataFrame): Test dataset.
    """
    _, test_df = train_test_split(data, test_size=0.20)
    [encoder, lb, lr_model] = pickle.load(open("model/lr_model.pkl", "rb"))

    X_test, y_test, _, _ = process_data(
        X=test_df,
        categorical_features=fake_categorical_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    preds = inference(lr_model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert precision >= 0.0
    assert precision <= 1.0

    assert recall >= 0.0
    assert recall <= 1.0

    assert fbeta >= 0.0
    assert fbeta <= 1.0
