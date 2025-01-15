import random
import torch
import matplotlib.pyplot as plt
from config import load_config
from data import get_data_loaders
from model import LinearMLP, NonLinearMLP, CNN
from helper import get_device
import os
from torchvision.transforms import Compose, Resize, RandomRotation, RandomHorizontalFlip, ToTensor

# Mapping of class indices to human-readable labels
classes = { 
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)', 
    2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 
    4: 'Speed limit (70km/h)', 
    5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 
    7: 'Speed limit (100km/h)', 
    8: 'Speed limit (120km/h)', 
    9: 'No passing', 
    10: 'No passing veh over 3.5 tons', 
    11: 'Right-of-way at intersection', 
    12: 'Priority road', 
    13: 'Yield', 
    14: 'Stop', 
    15: 'No vehicles', 
    16: 'Veh > 3.5 tons prohibited', 
    17: 'No entry', 
    18: 'General caution', 
    19: 'Dangerous curve left', 
    20: 'Dangerous curve right', 
    21: 'Double curve', 
    22: 'Bumpy road', 
    23: 'Slippery road', 
    24: 'Road narrows on the right', 
    25: 'Road work', 
    26: 'Traffic signals', 
    27: 'Pedestrians', 
    28: 'Children crossing', 
    29: 'Bicycles crossing', 
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing', 
    32: 'End speed + passing limits', 
    33: 'Turn right ahead', 
    34: 'Turn left ahead', 
    35: 'Ahead only', 
    36: 'Go straight or right', 
    37: 'Go straight or left', 
    38: 'Keep right', 
    39: 'Keep left', 
    40: 'Roundabout mandatory', 
    41: 'End of no passing', 
    42: 'End no passing veh > 3.5 tons'
}



def show_predictions(config_path, classes=classes):
    """
    Display predictions of the model alongside true labels.

    Args:
        config_path (str): Path to the configuration YAML file.
        classes (dict): Mapping of class indices to human-readable labels.
    """
    config = load_config(config_path)

    # Load original data loader (unmodified images)
    _, _, test_loader_original = get_data_loaders(config_path, original=True)

    # Load transformations for prediction
    transforms = Compose([
        RandomRotation(degrees=config["transforms"]["rotation"]),
        RandomHorizontalFlip(p=config["transforms"]["horizontal_flip"])
    ])

    # Random batch from the original data loader
    random_batch_idx = random.randint(0, len(test_loader_original) - 1)
    random_batch_original = list(test_loader_original)[random_batch_idx]
    original_images, labels = random_batch_original

    # Initialize model
    model_name = config["model"]["name"]
    path = config["training"]["model_path"]

    if model_name == "CNN":
        model = CNN(input_features=config["model"]["in_channels"], hidden_size=config["model"]["hidden_dim"], num_classes=config["model"]["num_classes"], dropout=config["model"]["dropout"])
    elif model_name == "LinearMLP":
        input_features = original_images[0].shape[0] * original_images[0].shape[1] * original_images[0].shape[2]
        model = LinearMLP(input_features=input_features, hidden_size=config["model"]["hidden_dim"], num_classes=config["model"]["num_classes"])
    elif model_name == "NonLinearMLP":
        input_features = original_images[0].shape[0] * original_images[0].shape[1] * original_images[0].shape[2]
        model = NonLinearMLP(input_features=input_features, hidden_size=config["model"]["hidden_dim"], num_classes=config["model"]["num_classes"], dropout=config["model"]["dropout"])
    else:
        raise ValueError("Unsupported model name")

    # Load model weights
    if not os.path.exists(path) or not any(f.endswith(".pth") for f in os.listdir(path)):
        raise ValueError("Model does not exist")
    model.load_state_dict(torch.load(os.path.join(path, os.listdir(path)[-1]), weights_only=True))
    model.to(get_device())
    model.eval()

    # Generate predictions
    test_labels_pred = []
    transformed_images = []

    with torch.inference_mode():
        for img in original_images:
            transformed_img = transforms(img)
            transformed_images.append(transformed_img)
            img = transformed_img.unsqueeze(0).to(get_device())
            pred = model(img)
            _, predicted_label = torch.max(pred, 1)
            test_labels_pred.append(predicted_label.item())

    # Plot original images with predictions
    fig = plt.figure(figsize=(9, 9))
    rows, cols = 3, 3
    for idx in range(rows * cols):
        orig_img = original_images[idx]
        label = classes[test_labels_pred[idx]]
        true_label = classes[labels[idx].item()]
        title_text = f"Pred: {label} | Truth: {true_label}"

        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.imshow(orig_img.permute(1, 2, 0))
        if label == true_label:
            ax.set_title(title_text, color="green", fontsize=7)
        else:
            ax.set_title(title_text, color="red", fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Show predictions for a random batch of images.
    show_predictions("configs/modelv0_param1.yaml")
