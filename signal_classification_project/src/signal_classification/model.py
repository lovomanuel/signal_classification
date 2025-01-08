import torch.nn as nn
from data import get_data_loaders

class TrafficSignClassifier(nn.Module):
    def __init__(self, input_features : int, hidden_size: int, num_classes: int):
        super().__init__()
        self.stack_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=num_classes)
        )
    def forward(self, x):
        return self.stack_layers(x)



