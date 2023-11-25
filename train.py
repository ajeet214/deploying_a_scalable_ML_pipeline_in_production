"""
train.py

This script serves as the entry point for training the machine learning model. It uses Hydra for configuration management,
loading the configuration from the specified path and triggering the training process by calling the 'training' function
from the 'train_model' module.

Usage:
- Run this script to initiate the model training with the provided configuration.

Note:
- Make sure Hydra is installed to support configuration management.

Author:
Ajeet Kumar Verma
"""
import hydra
from module.train_model import training
from omegaconf import DictConfig


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(config: DictConfig):
    # Entry point of the script using Hydra for configuration management.
    # It loads the configuration from the specified path and runs the training function.

    training(config)


if __name__ == "__main__":
    # Execute the main function when the script is run directly.
    main()