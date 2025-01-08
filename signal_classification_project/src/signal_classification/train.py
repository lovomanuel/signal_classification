import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm


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
    
def train(model, train_loader, val_loader, loss_function, optimizer, epochs, learning_rate):
    model.train()
    loss_function = get_loss_function(loss_function)
    optimizer = get_optimizer(optimizer, model.parameters(), learning_rate)

    for epoch in range(epochs):
        print(f"Epoch: {epoch}\n-------")
        
        ### Training
        model.train()  # Ensure model is in training mode
        train_loss = 0
        for batch, (images, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
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
            for batch, (images, labels) in enumerate(tqdm(val_loader)):
                output = model(images)
                loss = loss_function(output, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        print(f"Validation loss: {val_loss}")
        print(f"Validation accuracy: {val_accuracy:.2f}%")
        print("\n")



    

            







    
