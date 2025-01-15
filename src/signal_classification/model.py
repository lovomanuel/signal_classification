import torch.nn as nn
from config import load_config
import os
import torch
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

class LinearMLP(nn.Module):
    """
    Linear Multi-Layer Perceptron (MLP) model.

    Args:
        input_features (int): Number of input features.
        hidden_size (int): Number of hidden units in the single hidden layer.
        num_classes (int): Number of output classes.
    """

    def __init__(self, input_features: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.stack_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the LinearMLP model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output logits.
        """
        return self.stack_layers(x)


class NonLinearMLP(nn.Module):
    """
    Non-linear Multi-Layer Perceptron (MLP) model with ReLU and dropout.

    Args:
        input_features (int): Number of input features.
        hidden_size (int): Number of hidden units in the single hidden layer.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability.
    """

    def __init__(self, input_features: int, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.stack_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_size, out_features=num_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Forward pass of the NonLinearMLP model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output logits.
        """
        return self.stack_layers(x)


class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) model for image classification.

    Args:
        input_features (int): Number of input channels (e.g., 3 for RGB images).
        hidden_size (int): Number of filters in the first convolutional layer.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability.
    """

    def __init__(self, input_features: int, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_features, out_channels=hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size * 2, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * 2 * 7 * 7, 128),  # Assumes input image size of 28x28.
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm2d(hidden_size * 2)

    def forward(self, x):
        """
        Forward pass of the CNN model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output logits.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def model(config_path):
    """
    Load the specified model based on the configuration file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        None: Prints the model architecture and parameter count.
    """
    config = load_config(config_path)
    logger.info("Loaded configuration from %s", config_path)
    model_name = config["model"]["name"]
    PATH = "data/processed/"

    if model_name == "CNN":
        # Initialize CNN model.
        model = CNN(
            input_features=config["model"]["in_channels"],
            hidden_size=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        )
    elif model_name == "LinearMLP":
        # Calculate input features for LinearMLP.
        image_path = os.path.join(PATH, "Training", "train_img_0_31.pt")
        image = torch.load(image_path, weights_only=True)
        input_features = image[0].shape[0] * image[0].shape[1] * image[0].shape[2]
        model = LinearMLP(
            input_features=input_features,
            hidden_size=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
        )
    elif model_name == "NonLinearMLP":
        # Calculate input features for NonLinearMLP.
        image_path = os.path.join(PATH, "train", "train_img_0.pt")
        image = torch.load(image_path, weights_only=True)
        input_features = image[0].shape[0] * image[0].shape[1] * image[0].shape[2]
        model = NonLinearMLP(
            input_features=input_features,
            hidden_size=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        )
    else:
        logger.error("Unsupported model name: %s", model_name)
        raise ValueError("Unsupported model name")

    # Log model details.
    logger.info("Model architecture: %s", model)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    model(args.config)
