import torch.utils.data.dataloader
from train import get_loss_function
import torch
from tqdm import tqdm
import os
from data import get_data_loaders
from config import load_config
from model import LinearMLP, NonLinearMLP, CNN
from helper import get_device
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt


def evaluate(config_path):
    config = load_config(config_path)
    _, _, test_loader = get_data_loaders(config_path)

    los = config["loss"]["loss"]
    model_name = config["model"]["name"]
    path = config["training"]["model_path"]
    if model_name == "CNN":
        model = CNN(input_features=config["model"]["in_channels"], hidden_size=config["model"]["hidden_dim"], num_classes=config["model"]["num_classes"], dropout=config["model"]["dropout"])
        if not os.path.exists(path):
            raise ValueError("Model path does not exist")
        if os.path.exists(path) and any(f.endswith(".pth") for f in os.listdir(path)):
            model.load_state_dict(torch.load(os.path.join(path, os.listdir(path)[-3]), weights_only=True))
            #nel nome del pth metterai la loss e prendi quello con la loss piu bassa
        else:
            raise ValueError("Model does not exist")
    elif model_name == "LinearMLP":
        image = next(iter(test_loader))[0]
        input_features = image.shape[1] * image.shape[2] * image.shape[3]
        model = LinearMLP(input_features=input_features, hidden_size=config["model"]["hidden_dim"], num_classes=config["model"]["num_classes"])
        if not os.path.exists(path):
            raise ValueError("Model path does not exist")
        if os.path.exists(path) and any(f.endswith(".pth") for f in os.listdir(path)):
            model.load_state_dict(torch.load(os.path.join(path, os.listdir(path)[-1]), weights_only=True))
        else:
            raise ValueError("Model does not exist")
    elif model_name == "NonLinearMLP":
        image = next(iter(test_loader))[0]
        input_features = image.shape[1] * image.shape[2] * image.shape[3]
        model = NonLinearMLP(input_features=input_features, hidden_size=config["model"]["hidden_dim"], num_classes=config["model"]["num_classes"], dropout=config["model"]["dropout"])
        if not os.path.exists(path):
            raise ValueError("Model path does not exist")
        if os.path.exists(path) and any(f.endswith(".pth") for f in os.listdir(path)):
            model.load_state_dict(torch.load(os.path.join(path, os.listdir(path)[-1]), weights_only=True))
        else:
            raise ValueError("Model does not exist")
    else:
        raise ValueError("Unsupported model name")
    
    dev = get_device()
    model.eval()
    model.to(dev)
    loss_function = get_loss_function(los)
    test_loss = 0
    correct = 0
    total = 0

    num_classes = config["model"]["num_classes"]
    confusion_matrix = ConfusionMatrix(num_classes=num_classes, task='multiclass')

    # Evaluate model
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch, (images, y_true) in enumerate(tqdm(test_loader)):
            y_true = y_true.to(dev)
            images = images.to(dev)
            y_pred = model(images)
            loss = loss_function(y_pred, y_true)
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(y_pred, 1)
            total += y_true.size(0)
            correct += (predicted == y_true).sum().item()
            all_preds.append(predicted.cpu())
            all_labels.append(y_true.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    confusion_matrix.update(all_preds, all_labels)
    confusion_matrix = confusion_matrix.compute()
    fig, ax = plot_confusion_matrix(confusion_matrix.numpy(), figsize=(10, 10))

    #show confusion matrix
    plt.show()

    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": test_loss,
            "model_acc": test_accuracy}


