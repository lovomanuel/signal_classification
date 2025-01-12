import torch.nn as nn
from config import load_config
import os
import torch


class LinearMLP(nn.Module):
    def __init__(self, input_features : int, hidden_size: int, num_classes: int):
        super().__init__()
        self.stack_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=num_classes)
        )
    def forward(self, x):
        return self.stack_layers(x)
    
class NonLinearMLP(nn.Module):
    def __init__(self, input_features : int, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.stack_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_size, out_features=num_classes),
            nn.ReLU()
        )
    def forward(self, x):
        return self.stack_layers(x)

class CNN(nn.Module):
    def __init__(self, input_features : int, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels= input_features, out_channels= hidden_size, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels= hidden_size, out_channels= hidden_size*2, kernel_size = 5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * 2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm2d(hidden_size*2)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
def model(config_path):
    config = load_config(config_path)
    model_name = config["model"]["name"]
    PATH = "data/processed/"

    if model_name == "CNN":
        model = CNN(input_features=config["model"]["in_channels"], hidden_size=config["model"]["hidden_dim"], num_classes=config["model"]["num_classes"], dropout=config["model"]["dropout"])
    elif model_name == "LinearMLP":
        image_path = os.path.join(PATH, "train", "train_img_0.pt")
        image = torch.load(image_path, weights_only=True)
        input_features = image[0].shape[0] * image[0].shape[1] * image[0].shape[2]
        model = LinearMLP(input_features=input_features, hidden_size=config["model"]["hidden_dim"], num_classes=config["model"]["num_classes"])
    elif model_name == "NonLinearMLP":
        image_path = os.path.join(PATH, "train", "train_img_0.pt")
        image = torch.load(image_path, weights_only=True)   
        input_features = image[0].shape[0] * image[0].shape[1] * image[0].shape[2]
        model = NonLinearMLP(input_features=input_features, hidden_size=config["model"]["hidden_dim"], num_classes=config["model"]["num_classes"], dropout=config["model"]["dropout"])
    else:
        raise ValueError("Unsupported model name")
    
    print(f"Model architecture: {model}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    model("configs/modelv0_param1.yaml")


