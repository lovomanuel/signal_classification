import torch
import matplotlib.pyplot as plt
from config import load_config
from model import LinearMLP, NonLinearMLP, CNN
from helper import get_device
import os
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, Resize, ToTensor
from data import GTSRBDatasetRaw
import argparse
import random
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Mapping of class indices to human-readable labels
classes = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing veh over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing veh > 3.5 tons",
}

def show_predictions(config_path, classes=classes):
    """
    Display predictions of the model alongside true labels.

    Args:
        config_path (str): Path to the configuration YAML file.
        classes (dict): Mapping of class indices to human-readable labels.
    """
    logger.info("Loading configuration from %s", config_path)
    config = load_config(config_path)

    logger.info("Initializing dataset and transformations")
    test_dataset = GTSRBDatasetRaw("data/raw/", "test", Compose([Resize(config["transforms"]["resize"]), ToTensor()]))

    original_images = []
    true_labels = []
    transformed_images = []
    predicted_label = []

    transforms = Compose(
        [
            RandomRotation(degrees=config["transforms"]["rotation"]),
            RandomHorizontalFlip(p=config["transforms"]["horizontal_flip"]),
        ]
    )
    random_indices = random.sample(range(0, len(test_dataset)), config["data"]["batch_size"])

    logger.info("Applying transformations to sample images")
    for i in range(0, config["data"]["batch_size"]):
        numb = random_indices[i]
        original_images.append(test_dataset[numb][0])
        transformed_image = transforms(test_dataset[numb][0])
        transformed_images.append(transformed_image)
        true_labels.append(test_dataset[numb][1])

    # Initialize model
    model_name = config["model"]["name"]
    path = config["training"]["model_path"]

    logger.info("Initializing the %s model", model_name)
    if model_name == "CNN":
        model = CNN(
            input_features=config["model"]["in_channels"],
            hidden_size=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        )
    elif model_name == "LinearMLP":
        input_features = original_images[0].shape[0] * original_images[0].shape[1] * original_images[0].shape[2]
        model = LinearMLP(
            input_features=input_features,
            hidden_size=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
        )
    elif model_name == "NonLinearMLP":
        input_features = original_images[0].shape[0] * original_images[0].shape[1] * original_images[0].shape[2]
        model = NonLinearMLP(
            input_features=input_features,
            hidden_size=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        )
    else:
        logger.error("Unsupported model name: %s", model_name)
        raise ValueError("Unsupported model name")

    # Load model weights
    if not os.path.exists(path) or not any(f.endswith(".pth") for f in os.listdir(path)):
        logger.error("Model file does not exist in path: %s", path)
        raise ValueError("Model does not exist")
    model_file = os.path.join(path, os.listdir(path)[-1])
    logger.info("Loading model weights from %s", model_file)
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model.to(get_device())
    model.eval()

    logger.info("Generating predictions")
    with torch.inference_mode():
        for img in transformed_images:
            img = img.unsqueeze(0)
            img = img.to(get_device())
            output = model(img)
            _, pred = torch.max(output, 1)
            predicted_label.append(pred.item())

    # Plot original images with predictions
    logger.info("Plotting predictions")
    fig = plt.figure(figsize=(9, 9))
    rows, cols = 3, 3
    for idx in range(rows * cols):
        orig_img = original_images[idx]
        label = classes[predicted_label[idx]]
        true_label = classes[true_labels[idx].item()]
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    config_path = args.config
    show_predictions(config_path)
