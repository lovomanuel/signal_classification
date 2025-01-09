import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader
from tqdm.auto import tqdm
from data import get_data_loaders
from model import LinearMLP, NonLinearMLP, CNN
from config import load_config
from helper import get_device
import os


def get_loss_function(loss_name):
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Loss function '{loss_name}' not supported.")
    
def get_optimizer(optimizer_name, model_parameters, learning_rate):
    if optimizer_name == "adam":
        return optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name == "sgd":
        return optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

def train(config_path):
    config = load_config(config_path)
    train_loader, val_loader, _ = get_data_loaders(config_path)

    model_name = config["model"]["name"]
    opt = config["optimizer"]["optimizer"]
    los = config["loss"]["loss"]
    
    if model_name == "CNN":
        model = CNN(input_features=config["model"]["in_channels"], hidden_size=config["model"]["hidden_dim"], num_classes=config["model"]["num_classes"], dropout=config["model"]["dropout"])
    elif model_name == "LinearMLP":
        image = next(iter(train_loader))[0]
        input_features = image.shape[1] * image.shape[2] * image.shape[3]
        model = LinearMLP(input_features=input_features, hidden_size=config["model"]["hidden_dim"], num_classes=config["model"]["num_classes"])
    elif model_name == "NonLinearMLP":
        image = next(iter(train_loader))[0]
        input_features = image.shape[1] * image.shape[2] * image.shape[3]
        model = NonLinearMLP(input_features=input_features, hidden_size=config["model"]["hidden_dim"], num_classes=config["model"]["num_classes"], dropout=config["model"]["dropout"])
    else:
        raise ValueError("Unsupported model name")
    
    dev = get_device()
    model.to(dev)
    path = config["training"]["model_path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_path = os.path.join(
        path,
        f"{config['model']['name']}_hidden{config['model']['hidden_dim']}_lr{config['training']['lr']}.pth"
    )

    optimizer = get_optimizer(opt, model.parameters(), config["training"]["lr"])
    loss_function = get_loss_function(los)
    min_val_loss = float("inf")

    for epoch in range(config["training"]["epochs"]):
        print(f"Epoch: {epoch}\n-------")
        save_path = os.path.join(
        path,
        f"{config['model']['name']}_hidden{config['model']['hidden_dim']}_lr{config['training']['lr']}_epoch{epoch}.pth")
        
        ### Training
        model.train()  # Ensure model is in training mode
        train_loss = 0
        for batch, (images, y_true) in enumerate(tqdm(train_loader)):
            images = images.to(dev)
            y_true = y_true.to(dev)
            optimizer.zero_grad()
            y_pred = model(images)
            loss = loss_function(y_pred, y_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        print(f"Train loss: {train_loss}")
        
        ### Validation
        model.eval()  # Switch to evaluation mode
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient calculations for validation
            for batch, (images, y_true) in enumerate(tqdm(val_loader)):
                images = images.to(dev)
                y_true = y_true.to(dev)
                y_pred = model(images)
                loss = loss_function(y_pred, y_true)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(y_pred, 1)
                total += y_true.size(0)
                correct += (predicted == y_true).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        if val_loss < min_val_loss and config["training"]["save_model"]:
            print(f"Validation loss decreased from {min_val_loss:.4f} to {val_loss:.4f}. Saving model to {save_path}")
            min_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        print(f"Validation accuracy: {val_accuracy:.2f}%")
        print("\n")
    
    print("Training complete!")



    


    

            







    
