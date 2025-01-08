from train import get_loss_function
import torch
from tqdm import tqdm

def evaluate(model, test_loader, loss_function):
    model.eval()
    loss_function = get_loss_function(loss_function)
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch, (images, labels) in enumerate(tqdm(test_loader)):
            output = model(images)
            loss = loss_function(output, labels)
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print("\n")