import torch
from torchvision.transforms import Compose, ToTensor, Resize, RandomRotation, RandomHorizontalFlip
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import matplotlib.pyplot as plt
import os
from PIL import Image
from signal_classification.config import load_config
import requests
import logging
import zipfile
import shutil
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_data():
    """
    Download and extract the GTSRB training, test, and ground truth datasets.

    Downloads the datasets from specified URLs, extracts the contents, organizes
    the folder structure, and removes the downloaded ZIP files after extraction.

    The datasets are organized as follows:
    - Training data is saved in "data/raw/Training".
    - Test data is saved in "data/raw/Test/Images".
    - Ground truth data is extracted into "data/raw/Test/Images".

    Logs progress during each step.
    """
    training_url = (
        "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
    )
    test_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
    ground_truth_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"

    training_path = "data/raw/Training"
    test_path = "data/raw/Test"

    training_file = os.path.join(training_path, "Training.zip")
    test_file = os.path.join(test_path, "Test.zip")
    ground_truth_file = os.path.join(test_path, "Ground_Truth.zip")

    os.makedirs(training_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Download training data
    logger.info("Downloading training data...")
    response = requests.get(training_url, stream=True)
    with open(training_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    logger.info("Training data downloaded.")

    # Download test data
    logger.info("Downloading test data...")
    response = requests.get(test_url, stream=True)
    with open(test_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    logger.info("Test data downloaded.")

    # Download ground truth data
    logger.info("Downloading ground truth data...")
    response = requests.get(ground_truth_url, stream=True)
    with open(ground_truth_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    logger.info("Ground truth data downloaded.")

    # Extract training data
    logger.info("Extracting training data...")
    with zipfile.ZipFile(training_file, "r") as zip_ref:
        zip_ref.extractall(training_path)

    extracted_folder_training = os.path.join(training_path, "GTSRB")  # Assuming this is the extracted root folder
    images_folder_training = os.path.join(extracted_folder_training, "Final_Training", "Images")

    if not os.path.exists(training_path):
        os.makedirs(training_path)
    shutil.move(images_folder_training, training_path)
    logger.info(f"Training data extracted to '{training_path}' folder.")
    shutil.rmtree(extracted_folder_training)  # Remove the extracted root folder
    os.remove(training_file)  # Remove the ZIP file

    # Extract test data
    logger.info("Extracting test data...")
    with zipfile.ZipFile(test_file, "r") as zip_ref:
        zip_ref.extractall(test_path)

    extracted_folder_test = os.path.join(test_path, "GTSRB")  # Assuming this is the extracted root folder
    images_folder_test = os.path.join(extracted_folder_test, "Final_Test", "Images")

    if not os.path.exists(test_path):
        os.makedirs(test_path)
    shutil.move(images_folder_test, test_path)
    logger.info(f"Test data extracted to '{test_path}' folder.")
    shutil.rmtree(extracted_folder_test)  # Remove the extracted root folder
    os.remove(test_file)  # Remove the ZIP file

    # Extract ground truth data
    logger.info("Extracting ground truth data...")
    with zipfile.ZipFile(ground_truth_file, "r") as zip_ref:
        zip_ref.extractall(os.path.join(test_path, "Images"))

    os.remove(ground_truth_file)  # Remove the ZIP file
    logger.info(f"Ground truth data extracted to '{os.path.join(test_path, 'Images')}' folder.")

    logger.info("Data download and extraction complete.")


def assert_data():
    """
    Verify that the required dataset directories exist. If not, trigger the data download process.

    Checks for the presence of the "data/raw/Training" and "data/raw/Test" directories. If either
    directory is missing, it calls the `download_data` function to download and set up the datasets.

    Logs the status of the dataset directories.
    """
    training_path = "data/raw/Training"
    test_path = "data/raw/Test"
    if not os.path.exists(training_path) or not os.path.exists(test_path):
        logger.info("Data directories not found. Downloading data...")
        download_data()
    else:
        logger.info("Data directories found.")


class GTSRBDatasetRaw(Dataset):
    """
    Custom Dataset class for loading raw GTSRB data.

    Args:
        root (str): Path to the dataset root directory.
        split (str): Dataset split, one of "train", "test", or "val".
        transform (callable, optional): Transformations to apply to the images.

    Raises:
        ValueError: If the split is not "train", "test".
    """

    def __init__(self, root: str, split: str, transform=None):
        if split not in ["train", "test"]:
            raise ValueError('split must be either "train", "test"')
        self.root = root
        self.split = split
        self.transform = transform

        # Define the base folder based on the dataset split.
        base_folder = "Training/Images" if split == "train" else "Test/Images"
        self.labels = []
        self.data = []

        # Path to the directory containing the dataset.
        split_path = os.path.join(root, base_folder)
        if split == "train":
            for subfolder in os.listdir(split_path):  # Iterate through subfolders.
                subfolder_path = os.path.join(split_path, subfolder)
                if os.path.isdir(subfolder_path):  # Check if it is a directory.
                    label = subfolder[-2:]  # Extract the label from the folder name.
                    self.labels.append(label)  # Append the label to the list.
                    for img_name in os.listdir(subfolder_path):  # Iterate through image files.
                        if img_name.endswith(".ppm"):  # Only process .ppm files.
                            img_path = os.path.join(subfolder_path, img_name)
                            self.data.append((img_path, label))  # Append (image path, label) tuple.
        if split == "test":
            # Load test data and labels from CSV file.
            gt_path = os.path.join(split_path, "GT-final_test.csv")
            with open(gt_path, "r") as f:
                next(f)  # Skip the header row.
                for line in f:
                    # CSV format: Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
                    img_name, _, _, _, _, _, _, label = line.strip().split(";")
                    img_path = os.path.join(split_path, img_name)
                    self.data.append((img_path, label))

    def __len__(self):
        """
        Returns the size of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetch a sample (image, label) from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Tensor, Tensor]: Transformed image and its label.
        """
        img_path, label = self.data[idx]
        img = Image.open(img_path)  # Open the image.
        if self.transform:
            img = self.transform(img)  # Apply transformations.
        else:
            img = Resize((32, 32))(img)  # Resize the image to 32x32 pixels.
            img = ToTensor()(img)
        label = torch.tensor(int(label), dtype=torch.long)  # Convert label to tensor.
        return img, label


def save_processed_data(dataset, output_path, split):
    """
    Save preprocessed data to disk.

    Args:
        loader (Dataset): Dataset containing the samples to save.
        output_path (str): Directory to save the processed data.
        split (str): Dataset split name (e.g., "train", "test").
    """
    if split not in ["train", "test"]:
        raise ValueError('split must be either "train", "test"')
    base_folder = "Training" if split == "train" else "Test"
    split_path = os.path.join(output_path, base_folder)
    os.makedirs(split_path, exist_ok=True)  # Create the directory if it does not exist.
    logger.info(f"Saving {base_folder} data to '{split_path}' folder...")
    for idx in tqdm(range(len(dataset)), desc="Processing dataset"):
        image, label = dataset[idx]
        label_int = label.item()

        # save as png
        img_path = os.path.join(split_path, f"{split}_img_{idx}_{label_int}.png")
        if image.shape[0] == 1:  # Grayscale
            image = image.squeeze(0).numpy()  # Remove channel dimension
        else:  # RGB
            image = image.permute(1, 2, 0).numpy()  # Rearrange to (H, W, C)
        image = (image * 255).astype("uint8")
        image = Image.fromarray(image)
        image.save(img_path)
        logger.debug(f"Saved image {img_path}")
    logger.info(f"Processed {base_folder} data saved to '{split_path}' folder.")


class GTSRBDatasetProcessed(Dataset):
    """
    Custom Dataset class for loading preprocessed GTSRB data saved as .png files.

    This class is designed to work with preprocessed GTSRB (German Traffic Sign Recognition Benchmark) data.
    It loads images and their corresponding labels from a specified directory containing .png files. The dataset
    is divided into two splits: "train" and "test".

    Args:
        processed_path (str): The base directory where the processed data is stored.
        split (str): Specifies the dataset split to load. Must be either "train" or "test".

    Attributes:
        processed_path (str): Path to the directory containing the processed data.
        split (str): The dataset split being loaded ("train" or "test").
        data (List[str]): List of file paths for all .png files in the specified split.

    Raises:
        ValueError: If the `split` argument is not "train" or "test".

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Fetches the image and its label at the specified index.
    """

    def __init__(self, processed_path: str, split: str):
        if split not in ["train", "test"]:
            raise ValueError('split must be either "train" or "test"')
        self.processed_path = processed_path
        self.split = split
        self.data = []
        if split == "train":
            split_path = os.path.join(processed_path, "Training")
        else:
            split_path = os.path.join(processed_path, "Test")
        for file in os.listdir(split_path):
            if file.endswith(".png"):  # Only process .png files.
                file_path = os.path.join(split_path, file)
                self.data.append(file_path)

    def __len__(self):
        """
        Returns the size of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetch a preprocessed sample (image, label) from the dataset.

        The method retrieves an image and its corresponding label based on the specified index.
        The label is extracted from the filename, assuming the format includes the label as the
        last numeric part before the file extension (e.g., "split_img_0_3.png" -> label = 3).

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Tensor, Tensor]:
                - Tensor: Image loaded and transformed into a PyTorch tensor.
                - Tensor: Label as a long tensor.
        """
        file_path = self.data[idx]
        label = int(file_path.split("_")[-1].split(".")[0])  # Extract label from filename
        img = Image.open(file_path)  # Open the image file
        img = ToTensor()(img)  # Convert the image to a PyTorch tensor
        return img, torch.tensor(label, dtype=torch.long)


def dataLoader(config_path: str, save_processed: bool = False):
    """
    Load and preprocess GTSRB dataset, returning PyTorch DataLoaders.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
    """
    config = load_config(config_path)
    use_processed = config["data"]["use_processed"]

    if use_processed:
        # Load preprocessed data.
        logger.info("Loading preprocessed data...")
        train_data = GTSRBDatasetProcessed(config["data"]["processed_path"], "train")
        test_data = GTSRBDatasetProcessed(config["data"]["processed_path"], "test")
    else:
        # Process raw data.
        logger.info("Processing raw data...")
        transforms = Compose(
            [
                Resize(tuple(config["transforms"]["resize"])),
                RandomRotation(degrees=config["transforms"]["rotation"]),
                RandomHorizontalFlip(p=config["transforms"]["horizontal_flip"]),
                ToTensor(),
            ]
        )

        train_data = GTSRBDatasetRaw(config["data"]["raw_path"], "train", transform=transforms)
        test_data = GTSRBDatasetRaw(config["data"]["raw_path"], "test", transform=transforms)
        if save_processed:
            save_processed_data(train_data, config["data"]["processed_path"], "train")
            save_processed_data(test_data, config["data"]["processed_path"], "test")

    train_size = int(config["data"]["split_percentage"] * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    # Create DataLoaders.
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    logger.info("Training data loaded successfully.")
    logger.info(f"Training data size: {len(train_data)} samples subdivided in batches of {batch_size} images.")
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logger.info("Validation data loaded successfully.")
    logger.info(f"Validation data size: {len(val_data)} samples subdivided in batches of {batch_size} images.")
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logger.info("Test data loaded successfully.")
    logger.info(f"Test data size: {len(test_data)} samples subdivided in batches of {batch_size} images.")

    logger.info("Data loaded successfully.")

    return train_loader, val_loader, test_loader


def show_images(loader):
    """
    Display random images from the dataset with their labels.

    Args:
        loader (DataLoader): DataLoader object for the dataset.
        classes (dict): Mapping of class indices to human-readable labels.
    """
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
    random_batch_idx = random.randint(0, len(loader) - 1)
    random_batch = list(loader)[random_batch_idx]
    images, labels = random_batch

    fig = plt.figure(figsize=(9, 9))
    fig.suptitle("Random Images from the Dataset", fontsize=16)
    rows, cols = 4, 4
    for idx in range(rows * cols):
        random_idx = torch.randint(0, len(loader.dataset), size=[1]).item()
        img, label = loader.dataset[random_idx]
        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.imshow(img.permute(1, 2, 0))  # Convert image from CxHxW to HxWxC
        ax.set_title(classes[label.item()])
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    assert_data()
    config_path = "configs/data/data.yaml"
    config = load_config(config_path)
    transforms = Compose(
        [
            Resize(tuple(config["transforms"]["resize"])),
            RandomRotation(degrees=config["transforms"]["rotation"]),
            RandomHorizontalFlip(p=config["transforms"]["horizontal_flip"]),
            ToTensor(),
        ]
    )
    if not os.path.exists(config["transforms"]["processed_path"]):
        save_processed_data(GTSRBDatasetRaw(config["transforms"]["raw_path"], "train", transform=transforms), config["transforms"]["processed_path"], "train")
        save_processed_data(GTSRBDatasetRaw(config["transforms"]["raw_path"], "test", transform=transforms), config["transforms"]["processed_path"], "test")

if __name__ == "__main__":
    main()

