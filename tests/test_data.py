from torch.utils.data import Dataset
from signal_classification.data import *
from signal_classification.config import load_config

def test_download_data():
    assert_data()

def test_dataset_init():
    train_dataset = GTSRBDatasetRaw("data/raw", "train")
    test_dataset = GTSRBDatasetRaw("data/raw", "test")

    assert len(train_dataset) > 0, "Train dataset is empty"
    assert len(test_dataset) > 0, "Test dataset is empty"

    img, label = train_dataset[0]

    assert isinstance(img, torch.Tensor), "Image is not a PyTorch Tensor."
    assert img.ndim == 3, "Image does not have 3 dimensions (C, H, W)."
    assert isinstance(label, torch.Tensor), "Label is not a PyTorch Tensor."

def test_dataloader():
    train_loader, val_loader, test_loader = dataLoader("tests/test_config.yaml")

    assert isinstance(train_loader, torch.utils.data.DataLoader), "Train loader is not a DataLoader."
    assert isinstance(val_loader, torch.utils.data.DataLoader), "Validation loader is not a DataLoader."
    assert isinstance(test_loader, torch.utils.data.DataLoader), "Test loader is not a DataLoader."

    train_iter = iter(train_loader)
    img, label = next(train_iter)

    config = load_config("tests/test_config.yaml")

    batch_size = config["data"]["batch_size"]
    

    assert isinstance(img, torch.Tensor), "Image is not a PyTorch Tensor."
    assert img.ndim == batch_size, "Image does not have 4 dimensions (B, C, H, W)."
    assert isinstance(label, torch.Tensor), "Label is not a PyTorch Tensor."



