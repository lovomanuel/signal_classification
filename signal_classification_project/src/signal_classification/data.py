import torch
from torchvision.transforms import Compose, ToTensor, Resize, RandomRotation, RandomHorizontalFlip, ColorJitter, Normalize
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, random_split
import matplotlib.pyplot as plt
import os
from PIL import Image
from config import load_config

##aggiungi funzione iniziale per scaricare il dataset se non già scaricato

#capisci se devi salvare il file modificato o no

class GTSRBDatasetRaw(Dataset): #class with inheritance from Dataset class
    def __init__(self, root: str, split: str, transform=None): #constructor: root is the path to the dataset, split is the split of the dataset (train or test)
        if split not in ['train', 'test', 'val']:
            raise ValueError('split must be either "train" or "test" or "val"')
        self.root = root 
        self.split = split
        self.transform = transform

        base_folder = 'Training' if split == 'train' else 'Final_Test/Images'
        self.labels = []
        self.data = []

        #path to directory containing the dataset
        split_path = os.path.join(root, base_folder)
        if split == 'train':
            for subfolder in os.listdir(split_path): #for each subfolder in the directory
                subfolder_path = os.path.join(split_path, subfolder)
                if os.path.isdir(subfolder_path):  # Check if it's a directory
                    label = subfolder[-2:]  # Get the label from the folder name
                    self.labels.append(label) # Append the label to the labels list
                    for img_name in os.listdir(subfolder_path):
                        if img_name.endswith('.ppm'):
                            img_path = os.path.join(subfolder_path, img_name)
                            self.data.append((img_path, label)) # Append the image path and label to the data list
            #I could can include the part of opening image and transforming here but there could be some problems:
            #for example, by using getitem, I can open the image and transform it only when I need it, not before
            #this is useful when the dataset is too big and I don't want to load all the images at once
            #moreover, dataloader can load the images in parallel, speeding up the process
        if split == 'test':
            gt_path = os.path.join(split_path, 'GT-final_test.csv')
            with open(gt_path, 'r') as f:
                next(f)
                for line in f:
                    #Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
                    img_name, _, _, _, _, _, _, label = line.strip().split(';')
                    img_path = os.path.join(split_path, img_name)
                    self.data.append((img_path, label))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path) #The function Image.open() opens an image file from the specified file path (img_path) and creates an instance of PIL.Image.Image, which provides methods and attributes to manipulate the image (e.g., resizing, cropping, and format conversion). It does not load the entire image into memory immediately but instead opens a file pointer to access the image data. This deferred loading means the image is only fully read into memory when operations are performed or when it is converted to another format.
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(int(label), dtype=torch.long) 
        return img, label
    
class GTSRBDatasetProcessed(Dataset):
    def __init__(self, processed_path: str, split: str):
        if split not in ['train', 'test', 'val']:
            raise ValueError('split must be either "train" or "test" or "val"')
        self.processed_path = processed_path
        self.split = split
        self.data = []
        split_path = os.path.join(processed_path, split)
        for file in os.listdir(split_path):
            if file.endswith('.pt'):
                file_path = os.path.join(split_path, file)
                self.data.append(file_path)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        img, label = torch.load(file_path, weights_only=True)
        return img, label
    
# Save preprocessed data to disk
def save_processed_data(loader, output_path, split):
    split_path = os.path.join(output_path, split)
    os.makedirs(split_path, exist_ok=True)
    for idx, (img, label) in enumerate(loader):
        img = img.squeeze(0)
        label = label.squeeze(0)
        file_path = os.path.join(split_path, f"{split}_img_{idx}.pt")
        torch.save((img, label), file_path)
    print(f"Saved {split} data to {split_path}")

    
# Main function to get DataLoaders
def get_data_loaders(config_path: str):
    config = load_config(config_path)
    use_processed = config["data"]["use_processed"]

    if use_processed:
        print("Loading preprocessed data...")
        train_data = GTSRBDatasetProcessed(config["data"]["processed_path"], "train")
        val_data = GTSRBDatasetProcessed(config["data"]["processed_path"], "val")
        test_data = GTSRBDatasetProcessed(config["data"]["processed_path"], "test")

    else:
        print("Processing raw data...")
        # Define transformations
        transforms = Compose([
            Resize(tuple(config["transforms"]["resize"])),
            RandomRotation(degrees=config["transforms"]["rotation"]),
            RandomHorizontalFlip(p=config["transforms"]["horizontal_flip"]),
            ToTensor()
        ])
        train_data = GTSRBDatasetRaw(config["data"]["raw_path"], "train", transform=transforms)
        test_data = GTSRBDatasetRaw(config["data"]["raw_path"], "test", transform=transforms)

        # Train/Validation Split
        train_size = int(config["data"]["split_percentage"] * len(train_data))
        val_size = len(train_data) - train_size
        train_data, val_data = random_split(train_data, [train_size, val_size])

        # Save preprocessed data if needed
        save_processed_data(DataLoader(train_data, batch_size=1, shuffle=False), config["data"]["processed_path"], "train")
        save_processed_data(DataLoader(val_data, batch_size=1, shuffle=False), config["data"]["processed_path"], "val")
        save_processed_data(DataLoader(test_data, batch_size=1, shuffle=False), config["data"]["processed_path"], "test")

    # Create DataLoaders
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


#A DataLoader in PyTorch is an iterator that allows you to iterate over the dataset in batches, making it efficient for training and evaluation. For each iteration, it retrieves a batch of images and labels, both in the form of tensors.
#The images are returned as a tensor of shape (batch_size, channels, height, width)
#The labels are returned as a tensor of shape (batch_size, _)

#sistema questa funzione per mostrare le immagini









