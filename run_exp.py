'''
    This class will parse the yaml file in the root directory and will
        1. Extract options for execution of a NN model
        2. Execute the experiment with the provided options
'''
import ModelRepository as mr
import train_net
import Inference
import CreateAttacks
import yaml

yml_location = r".\config.yaml"

if __name__ == '__main__':     
    with open(yml_location, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        print("selected task...", data_loaded["task"])
        if(data_loaded["task"]["type"].lower() == "training"):
            print("Initiating training on the selected network...")
            train_net.initiate_traning(data_loaded)
        elif(data_loaded["task"]["type"].lower() == "inference"):
            print("Performing inference on the selected model...")
            Inference.inference_net.initiate_inference(data_loaded)
        elif(data_loaded["task"]["type"].lower()== "attack"):
            print("Creating adversarial examples based on the selected algorithm...")
            CreateAttacks.attack_net.initiate_attack_creation(data_loaded)
    
