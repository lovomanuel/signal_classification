from data import get_data_loaders
from train import train
from evaluate import evaluate
from visualize import show_images
from visualize import show_predictions

#I want to as hyperparam in command line data_raw as boolean 


if __name__ == "__main__":
    config_path = "configs/modelv2_param1.yaml"
    
    show_predictions(config_path)
    