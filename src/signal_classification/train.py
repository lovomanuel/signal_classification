import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader
from tqdm.auto import tqdm
from data import dataLoader
from model import LinearMLP, NonLinearMLP, CNN
from config import load_config
from helper import get_device
import os
import argparse


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


def train(config_path):
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
        # Determine input features based on data sample dimensions.
        image = next(iter(train_loader))[0]
        input_features = image.shape[1] * image.shape[2] * image.shape[3]
        model = LinearMLP(
            input_features=input_features,
            hidden_size=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
        )
    elif model_name == "NonLinearMLP":
        # Determine input features based on data sample dimensions.
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

    # Prepare the device and model saving path.
    dev = get_device()  # Get CPU or GPU.
    model.to(dev)  # Move model to the device.
    path = config["training"]["model_path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_path = os.path.join(
        path, f"{config['model']['name']}_hidden{config['model']['hidden_dim']}_lr{config['training']['lr']}.pth"
    )

    # Initialize optimizer and loss function.
    optimizer = get_optimizer(opt, model.parameters(), config["training"]["lr"])
    loss_function = get_loss_function(los)
    min_val_loss = float("inf")

    # Training loop.
    for epoch in range(config["training"]["epochs"]):
        print(f"Epoch: {epoch}\n-------")
        save_path = os.path.join(
            path,
            f"{config['model']['name']}_hidden{config['model']['hidden_dim']}_lr{config['training']['lr']}_epoch{epoch}.pth",
        )

        ### Training
        model.train()  # Ensure the model is in training mode.
        train_loss = 0
        for batch, (images, y_true) in enumerate(tqdm(train_loader)):
            images, y_true = images.to(dev), y_true.to(dev)  # Move data to device.
            optimizer.zero_grad()  # Reset gradients.
            y_pred = model(images)  # Forward pass.
            loss = loss_function(y_pred, y_true)  # Compute loss.
            loss.backward()  # Backward pass.
            optimizer.step()  # Update weights.
            train_loss += loss.item()  # Accumulate training loss.

        train_loss /= len(train_loader)  # Average training loss.
        print(f"Train loss: {train_loss}")

        ### Validation
        model.eval()  # Set the model to evaluation mode.
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradients for validation.
            for batch, (images, y_true) in enumerate(tqdm(val_loader)):
                images, y_true = images.to(dev), y_true.to(dev)  # Move data to device.
                y_pred = model(images)  # Forward pass.
                loss = loss_function(y_pred, y_true)  # Compute loss.
                val_loss += loss.item()  # Accumulate validation loss.

                # Calculate accuracy.
                _, predicted = torch.max(y_pred, 1)
                total += y_true.size(0)
                correct += (predicted == y_true).sum().item()

        val_loss /= len(val_loader)  # Average validation loss.
        val_accuracy = 100 * correct / total  # Calculate accuracy percentage.

        # Save the model if validation loss improves.
        if val_loss < min_val_loss and config["training"]["save_model"]:
            print(f"Validation loss decreased from {min_val_loss:.4f} to {val_loss:.4f}. Saving model to {save_path}")
            min_val_loss = val_loss
            # elimina il precedente contenuto della cartella
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))
            torch.save(model.state_dict(), save_path)

        print(f"Validation accuracy: {val_accuracy:.2f}%")
        print("\n")

    print("Training complete!")


if __name__ == "__main__":
    # Entry point for script execution.
    parser = argparse.ArgumentParser(description="Train a model for signal classification.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    train(args.config)
