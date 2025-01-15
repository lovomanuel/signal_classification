import torch.utils.data.dataloader
from train import get_loss_function
import torch
from tqdm import tqdm
import os
from data import dataLoader
from config import load_config
from model import LinearMLP, NonLinearMLP, CNN  # Import model classes.
from helper import get_device  # Utility function to get the device (CPU/GPU).
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

def evaluate(config_path):
    """
    Evaluate a trained model on the test dataset.

    Args:
        config_path (str): Path to the configuration YAML file.

    The function:
    - Loads the model and test data based on configuration.
    - Computes the test loss and accuracy.
    - Prints the results.

    Raises:
        ValueError: If the model path or trained model file is missing.
    """
    config = load_config(config_path)  # Load configuration.
    logger.info("Loaded configuration from %s", config_path)
    _, _, test_loader = dataLoader(config_path)  # Get test DataLoader.

    # Extract configurations for model, loss, and saved model path.
    los = config["loss"]["loss"]
    model_name = config["model"]["name"]
    path = config["training"]["model_path"]

    # Initialize the model based on the specified type.
    logger.info("Initializing the %s model", model_name)
    if model_name == "CNN":
        model = CNN(
            input_features=config["model"]["in_channels"],
            hidden_size=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        )
    elif model_name == "LinearMLP":
        # Calculate input features for LinearMLP based on a sample image.
        image = next(iter(test_loader))[0]
        input_features = image.shape[1] * image.shape[2] * image.shape[3]
        model = LinearMLP(
            input_features=input_features,
            hidden_size=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
        )
    elif model_name == "NonLinearMLP":
        # Calculate input features for NonLinearMLP based on a sample image.
        image = next(iter(test_loader))[0]
        input_features = image.shape[1] * image.shape[2] * image.shape[3]
        model = NonLinearMLP(
            input_features=input_features,
            hidden_size=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        )
    else:
        logger.error("Unsupported model name: %s", model_name)
        raise ValueError("Unsupported model name")

    # Check if the model path and files exist.
    if not os.path.exists(path):
        logger.error("Model path does not exist: %s", path)
        raise ValueError("Model path does not exist")
    if os.path.exists(path) and any(f.endswith(".pth") for f in os.listdir(path)):
        model_file = os.path.join(path, os.listdir(path)[-1])
        logger.info("Loading model weights from %s", model_file)
        model.load_state_dict(torch.load(model_file, weights_only=True))
    else:
        logger.error("Model file does not exist in path: %s", path)
        raise ValueError("Model file does not exist")

    # Prepare for evaluation.
    dev = get_device()  # Determine device (CPU/GPU).
    model.eval()  # Set the model to evaluation mode.
    model.to(dev)  # Move model to the device.
    loss_function = get_loss_function(los)  # Load the loss function.
    logger.info("Model is ready for evaluation")
    test_loss = 0
    correct = 0
    total = 0

    # Evaluate the model on the test dataset.
    logger.info("Starting evaluation on the test dataset")
    all_preds = []
    all_labels = []
    with torch.inference_mode():  # Disable gradient computation for inference.
        for batch, (images, y_true) in enumerate(tqdm(test_loader)):  # Iterate through test batches.
            y_true = y_true.to(dev)  # Move labels to device.
            images = images.to(dev)  # Move images to device.
            y_pred = model(images)  # Forward pass.
            loss = loss_function(y_pred, y_true)  # Compute loss.
            test_loss += loss.item()  # Accumulate test loss.

            # Compute predictions and accuracy.
            _, predicted = torch.max(y_pred, 1)  # Get predicted class indices.
            total += y_true.size(0)  # Count total samples.
            correct += (predicted == y_true).sum().item()  # Count correct predictions.
            all_preds.append(predicted.cpu())  # Collect predictions.
            all_labels.append(y_true.cpu())  # Collect true labels.

    # Concatenate predictions and labels for metrics.
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute average loss and accuracy.
    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    # Store and print results.
    results = {"model_name": model.__class__.__name__, "model_loss": test_loss, "model_acc": test_accuracy}
    logger.info("Model: %s", results['model_name'])
    logger.info("Loss: %.4f", results['model_loss'])
    logger.info("Accuracy: %.2f%%", results['model_acc'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    evaluate(args.config)
