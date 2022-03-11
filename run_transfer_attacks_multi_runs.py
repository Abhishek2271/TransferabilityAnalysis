'''
    This class will parse the yaml file in the root directory and will
        1. Extract options for execution of a NN model
        2. Execute the experiment with the provided options

    Not really cleanest code but just for research purpose this works...
'''
import CreateAttacks
import os
import yaml
from CreateAttacks.attack_net import initiate_attack_transfer
from tensorpack import *
import logging
import pandas as pd
import numpy as np
import visualize_data 

yml_location = r"C:\Users\sab\Downloads\AI Testing\Source\Dorefanet\tensorpack\FullPrecisionModels\config_transfer_attack_multi_runs.yaml"

if __name__ == '__main__':     
    with open(yml_location, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        #get source data
        source_data = data_loaded["attack-options"]["source-data"]
        #get target model
        target_base_model = data_loaded["attack-options"]["target-base-model"]
        
        #get all sources (npz images crafted at source)
        sources = data_loaded["attack-options"]["sources"]
        print(sources)
        #get all targets (target model checkpoints, will be loaded to the target model)
        targets = data_loaded["attack-options"]["targets"]
        #setup log name
        dataset_name_log = source_data        

        #save complete resutls. 
        # Each element of this array is a adv_robustness_complete array i.e. each element is a complete result of a single run
        complete_results = []


        #For individual "source" is a different set of adversarial examples set. 
        # Each "set" is of sources contain examples generated from FP, 1, 2, 4, 8, 12, 16 bit networks. So 3 sets mean 3 x (FP, 1, 2, 4, 8, 12, 16)
        # In the end the results are averaged from each set 
        #Each set gives a complete transferability summary which contains transferability data from each to every other net
        for source in sources:
            #setup logger to log progress and the results
            trans_logger = logging.getLogger("trans_logger_{}".format(source))
            trans_logger.setLevel(logging.INFO)
            fh = logging.FileHandler(os.path.join('logs','transfer_log', "Transfer_data_{}.log".format(source)))
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s %(message)s')
            fh.setFormatter(formatter)
            trans_logger.addHandler(fh)

            #a (pandas dataframe) with complete summary of transferabiltiy results for each run (each run is a set of sources (FP --> FP, 1, 4, 8, 12, 16)). 
            # Is reset on every set of sources        
            adv_robustness_complete = []

            #Now, for each sources (FP, 1, 2, 4, 8, 12, 16) within this set 
            for source_images in sources[source]: 
                source_img = sources[source][source_images]
                source_log_name = os.path.splitext(os.path.basename(source_img))[0] 
                trans_logger.info("######## Source Image: {} Source location:  {}".format(source_log_name, source_img))        
                #for each row of transfer store the results (eg: FP --> FP, 1, 4, 8, 12, 16)                
                adv_robustness = []
                #Transfer the adv examples from this source to all target nets (FP, 1, 2, 4, 8, 12, 16)
                for current_target in targets:
                    bitwidth = targets[current_target]["bitwidth"]
                    target_model = targets[current_target]["modeldata"] 
                    if(bitwidth is None):
                        log_name = "FP"
                    else:
                        log_name = bitwidth 
                    logger.set_logger_dir(os.path.join('logs','transfer_log','transfer_log_run_{}'.format(source),'from-{}-transfer-'.format(source_log_name), '{}-{}'.format(dataset_name_log, log_name)))
                    trans_logger.info("--- Target BitWidth {}; Target model:  {}".format(log_name, target_model))
                    
                    adv_acc, transfer_rate, saved_adv_corr_images, saved_adv_false_images, agvl2_f, avglinf_f = initiate_attack_transfer(source_data, target_base_model, bitwidth, target_model, source_img)
                    
                    trans_logger.info("Adversarial accuracy of the target network: {}, Transfer rate: {} \n".format(adv_acc, transfer_rate))
                    agvl2 = float("{0:.4f}".format(agvl2_f))
                    avglinf = float("{0:.4f}".format(avglinf_f))                
                    
                    trans_logger.info("Average l2 distance of the incorrectly classified images: {}, Average l_inf distance of the incorrectly classified images: {} \n\n\n".format(agvl2, avglinf))
                    adv_robustness.append(adv_acc)
                print(adv_robustness)            
                trans_logger.info("Finished for {}. Starting next source.\n\n\n".format(source_log_name))
                print(target_model)
                adv_robustness_complete.append(adv_robustness)            
            print(adv_robustness_complete)
            trans_logger.info("Complete transfer summary: \n {}".format(adv_robustness_complete)) 
            complete_results.append(np.array(adv_robustness_complete)) 

        #find the average of all runs in the end
        avg = np.mean(complete_results, axis=0)
        print(avg.shape)
        print(avg.tolist())

        trans_logger.info("Average of all runs: \n {}".format(avg.tolist())) 
        #df = pd.DataFrame(adv_robustness_complete)
        #visualize_data.plot_data_points(df)

        