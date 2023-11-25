"""
Census FastAPI Application

This FastAPI application serves as an interface for model inference in the Census project.
It defines two endpoints: a root endpoint that displays a welcome message and an '/infer'
endpoint that accepts input data, performs model inference, and returns the predicted output.

The application uses Pydantic models for input validation and Hydra for configuration management.
"""
from typing import Dict
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import hydra

from module.data import process_data
from module.model import inference

# Create a FastAPI instance
app = FastAPI()


class CensusInputData(BaseModel):
    """
    Pydantic model for input data in the Census application.
    """
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 23,
                    "workclass": "Self-emp-not-inc",
                    "fnlgt": 8071,
                    "education": "HS-grad",
                    "education-num": 9,
                    "marital-status": "Married-civ-spouse",
                    "occupation": "Exec-managerial",
                    "relationship": "Husband",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "hours-per-week": 45,
                    "native-country": "United-States"
                }
            ]
        }
    }

# Define the root endpoint with a welcome message
@app.get(path="/")
def welcome_root():
    """
    Welcome message at the root endpoint.
    """
    return {"message": "Welcome to the project!"}

# Define the /infer endpoint for model inference
@app.post(path="/infer")
# @hydra.main(config_path=".", config_name="config", version_base="1.2")
async def prediction(input_data: CensusInputData) -> Dict[str, str]:
    """
    Perform model inference based on input data.

    Args:
        input_data (CensusInputData): Input data for model inference.

    Returns:
        dict: Dictionary containing the model output.
    """
    # Load model and configuration
    with hydra.initialize(config_path=".", version_base="1.2"):
        config = hydra.compose(config_name="config")
    [encoder, lb, model] = pickle.load(
        open(config["main"]["model_path"], "rb"))

    # Convert input data to DataFrame
    input_df = pd.DataFrame(
        {k: v for k, v in input_data.dict(by_alias=True).items()}, index=[0]
    )

    # Process input data for model inference
    processed_input_data, _, _, _ = process_data(
        X=input_df,
        categorical_features=config['main']['cat_features'],
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Perform model inference
    pred = inference(model, processed_input_data)

    # Return the model output
    return {"Output": ">50K" if int(pred[0]) == 1 else "<=50K"}


# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="main:app", host="0.0.0.0", port=5000)