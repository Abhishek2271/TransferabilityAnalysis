'''
    This class will parse the yaml file in the root directory and will
        1. Extract options for execution of a NN model
        2. Execute the experiment with the provided options
'''
import CreateAttacks
import os
import yaml
from CreateAttacks.attack_net import initiate_attack_transfer
from tensorpack import *
import logging
import pandas as pd
import visualize_data 

yml_location = r"C:\Users\sab\Downloads\AI Testing\Source\Dorefanet\tensorpack\FullPrecisionModels\config_transfer_attack.yaml"

if __name__ == '__main__':     
    with open(yml_location, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        #get source data
        source_data = data_loaded["attack-options"]["source-data"]
        #get target model
        target_base_model = data_loaded["attack-options"]["target-base-model"]
        
        #get all sources (npz images crafted at source)
        sources = data_loaded["attack-options"]["sources"]
        #get all targets (target model checkpoints, will be loaded to the target model)
        targets = data_loaded["attack-options"]["targets"]

        #setup logger
        dataset_name_log = source_data
        

        trans_logger = logging.getLogger("trans_logger")
        trans_logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join('logs','transfer_log', "Transfer_data.log"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        fh.setFormatter(formatter)
        trans_logger.addHandler(fh)

        #save complete data 
        adv_robustness_complete = []

        for current_source in sources:            
            source_img = sources[current_source]
            source_log_name = os.path.splitext(os.path.basename(source_img))[0] 
            trans_logger.info("######## Source Image: {} Source location:  {}".format(source_log_name, source_img))        
            #for each row of transfer store the results (eg: FP --> FP, 1, 4, 8, 12, 16)
            adv_robustness = []
            for current_target in targets:
                bitwidth = targets[current_target]["bitwidth"]
                target_model = targets[current_target]["modeldata"] 
                if(bitwidth is None):
                    log_name = "FP"
                else:
                    log_name = bitwidth 
                logger.set_logger_dir(os.path.join('logs','transfer_log','from-{}-transfer-'.format(source_log_name), '{}-{}'.format(dataset_name_log, log_name)))
                trans_logger.info("--- Target BitWidth {}; Target model:  {}".format(log_name, target_model))
                adv_acc, transfer_rate, saved_adv_corr_images, saved_adv_false_images, agvl2_f, avglinf_f = initiate_attack_transfer(source_data, target_base_model, bitwidth, target_model, source_img)
                trans_logger.info("Adversarial accuracy of the target network: {}, Transfer rate: {} \n".format(adv_acc, transfer_rate))
                agvl2 = float("{0:.4f}".format(agvl2_f))
                avglinf = float("{0:.4f}".format(avglinf_f))                
                trans_logger.info("Average l2 distance of the incorrectly classified images: {}, Average l_inf distance of the incorrectly classified images: {} \n\n\n".format(agvl2, avglinf))
                adv_robustness.append(adv_acc)
            print(adv_robustness)            
            trans_logger.info("Finished for {}. Starting next source.\n\n\n".format(source_log_name))
            adv_robustness_complete.append(adv_robustness)            
        print(adv_robustness_complete)
        trans_logger.info("Complete transfer summary: \n {}".format(adv_robustness_complete))      
       
        #df = pd.DataFrame(adv_robustness_complete)
        #visualize_data.plot_data_points(df)

        