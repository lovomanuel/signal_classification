from data import get_data_loaders
from train import train
from evaluate import evaluate

#I want to as hyperparam in command line data_raw as boolean 


if __name__ == "__main__":
    config_path = "configs/modelv2_param1.yaml"
    
    model_result = evaluate(config_path)
    print(model_result)
    