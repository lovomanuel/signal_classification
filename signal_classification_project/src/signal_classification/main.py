from data import get_data_loaders
from model import TrafficSignClassifier
from train import train
from evaluate import evaluate
import torch


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64, split_percentage=0.8)
    model = TrafficSignClassifier(input_features=3072, hidden_size=128, num_classes=43)
    loss_function = "cross_entropy"
    optimizer = "adam"
    epochs = 10
    learning_rate = 0.001

    train(model, train_loader, val_loader, loss_function, optimizer, epochs, learning_rate)
    evaluate(model, test_loader, loss_function)