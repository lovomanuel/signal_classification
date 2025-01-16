import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader
from tqdm import tqdm
from data import dataLoader
from model import LinearMLP, NonLinearMLP, CNN
from config import load_config
from helper import get_device
import os
import argparse
import logging
import wandb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def get_loss_function(loss_name):
    """
    Get the loss function based on its name.

    Args:
        loss_name (str): Name of the loss function ("cross_entropy" or "mse").

    Returns:
        nn.Module: Corresponding PyTorch loss function.

    Raises:
        ValueError: If the loss function name is unsupported.
    """
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Loss function '{loss_name}' not supported.")

def get_optimizer(optimizer_name, model_parameters, learning_rate):
    """
    Get the optimizer based on its name.

    Args:
        optimizer_name (str): Name of the optimizer ("adam" or "sgd").
        model_parameters (iterable): Model parameters to optimize.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        optim.Optimizer: Corresponding PyTorch optimizer.

    Raises:
        ValueError: If the optimizer name is unsupported.
    """
    if optimizer_name == "adam":
        return optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name == "sgd":
        return optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

def train(config_path, api_key):
    """
    Train a model based on the configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    The function:
    - Loads the model, data loaders, optimizer, and loss function.
    - Trains the model for a specified number of epochs.
    - Validates the model and saves the best-performing version.
    """
    # Load configuration and data loaders.
    config = load_config(config_path)
    logger.info("Loaded configuration from %s", config_path)
    train_loader, val_loader, _ = dataLoader(config_path)

    # Initialize the model.
    model_name = config["model"]["name"]
    opt = config["optimizer"]["optimizer"]
    los = config["loss"]["loss"]

    if model_name == "CNN":
        model = CNN(
            input_features=config["model"]["in_channels"],
            hidden_size=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        )
    elif model_name == "LinearMLP":
        image = next(iter(train_loader))[0]
        input_features = image.shape[1] * image.shape[2] * image.shape[3]
        model = LinearMLP(
            input_features=input_features,
            hidden_size=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
        )
    elif model_name == "NonLinearMLP":
        image = next(iter(train_loader))[0]
        input_features = image.shape[1] * image.shape[2] * image.shape[3]
        model = NonLinearMLP(
            input_features=input_features,
            hidden_size=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        )
    else:
        raise ValueError("Unsupported model name")

    logger.info("Initialized %s model", model_name)


    # Initialize WandB
    wandb.login(key=api_key)
    wandb.init(project="signal-classification", config={"model": model_name, "optimizer": opt, "loss": los, "lr": config["training"]["lr"], "epochs": config["training"]["epochs"]})

    # Prepare the device and model saving path.
    dev = get_device()
    model.to(dev)
    path = config["training"]["model_path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_path = os.path.join(
        path, f"{config['model']['name']}_hidden{config['model']['hidden_dim']}_lr{config['training']['lr']}.pth"
    )

    # Initialize optimizer and loss function.
    optimizer = get_optimizer(opt, model.parameters(), config["training"]["lr"])
    loss_function = get_loss_function(los)
    min_val_loss = float("inf")
    best_model_path = None
    max_val_accuracy = 0

    # Training loop.
    # Training loop
    for epoch in range(config["training"]["epochs"]):
        logger.info("Starting epoch %d", epoch)
        model.train()
        train_loss = 0

        for batch, (images, y_true) in enumerate(tqdm(train_loader)):
            images, y_true = images.to(dev), y_true.to(dev)
            optimizer.zero_grad()
            y_pred = model(images)
            loss = loss_function(y_pred, y_true)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        logger.info("Epoch %d: Train loss = %.4f", epoch, train_loss)

        ### Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch, (images, y_true) in enumerate(tqdm(val_loader)):
                images, y_true = images.to(dev), y_true.to(dev)
                y_pred = model(images)
                loss = loss_function(y_pred, y_true)
                val_loss += loss.item()

                _, predicted = torch.max(y_pred, 1)
                total += y_true.size(0)
                correct += (predicted == y_true).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        # Log metrics to WandB
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        # Save the best model
        if val_loss < min_val_loss:
            logger.info(
                "Validation loss improved from %.4f to %.4f. Saving model to %s",
                min_val_loss,
                val_loss,
                path,
            )
            min_val_loss = val_loss
            best_model_path = os.path.join(
                path,
                f"{model_name}_epoch{epoch}_val_loss{val_loss:.4f}.pth"
            )
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))
            torch.save(model.state_dict(), best_model_path)
        
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy

    # Upload best model to WandB as an artifact
    if best_model_path:
        name_of_the_model = "Best_model" + str(config["model"]["name"])
        artifact = wandb.Artifact(name_of_the_model, type="model", metadata={"accuracy": max_val_accuracy, "loss": min_val_loss})
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)

    logger.info("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for signal classification.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    parser.add_argument("--api_key", type=str, help="WandB API key.")
    args = parser.parse_args()
    train(args.config, args.api_key)
