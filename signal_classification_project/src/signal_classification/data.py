import torch
from torchvision.transforms import Compose, ToTensor, Resize, RandomRotation, RandomHorizontalFlip, ColorJitter, Normalize
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, random_split
import matplotlib.pyplot as plt
import os
from PIL import Image

##aggiungi funzione iniziale per scaricare il dataset se non già scaricato

#capisci se devi salvare il file modificato o no

PATH = "data/raw/gtsrb/GTSRB" ##uniforma questo in modo che da qualsiasi parte io lo runni funzioni

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }


transforms = Compose([
    Resize((32, 32)),  # Resize to the target size
    RandomRotation(degrees=15),  # Randomly rotate images by up to 15 degrees
    RandomHorizontalFlip(p=0.5),  # Randomly flip the images horizontally  # Apply random changes to brightness, contrast, etc.
    ToTensor()  # Convert to tensor
])

class GTSRBDataset(Dataset): #class with inheritance from Dataset class
    def __init__(self, root: str, split: str, transform=None): #constructor: root is the path to the dataset, split is the split of the dataset (train or test)
        if split not in ['train', 'test']:
            raise ValueError('split must be either "train" or "test"')
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
    
def get_data_loaders(batch_size: int, split_percentage: float):
    train_data = GTSRBDataset(root=PATH, split='train', transform=transforms)
    test_data = GTSRBDataset(root=PATH, split='test', transform=transforms)

    train_size = int(split_percentage * len(train_data))  # 80% for training
    val_size = len(train_data) - train_size  # 20% for validation

    # Split the dataset
    train_data, val_data = random_split(train_data, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, val_loader

#A DataLoader in PyTorch is an iterator that allows you to iterate over the dataset in batches, making it efficient for training and evaluation. For each iteration, it retrieves a batch of images and labels, both in the form of tensors.
#The images are returned as a tensor of shape (batch_size, channels, height, width)
#The labels are returned as a tensor of shape (batch_size, _)

#sistema questa funzione per mostrare le immagini

def show_images(loader, classes):
    #take a random loader 
    dataiter = iter(loader)
    images, labels = next(dataiter)
    fig = plt.figure(figsize=(9, 9))
    #fig name
    fig.suptitle('Random Images from the Dataset', fontsize=16)
    rows, cols = 4, 4
    for idx in range(rows * cols):
        random_idx = torch.randint(0, len(loader.dataset), size=[1]).item()
        img, label = loader.dataset[random_idx]
        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(classes[label.item()])
        ax.axis('off')
    plt.tight_layout()
    plt.show()







