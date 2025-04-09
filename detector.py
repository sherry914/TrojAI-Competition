# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

from tqdm import tqdm

from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, \
    load_models_dirpath
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
)


import torch
import torch.nn.functional as F
import helper as helper
import extractor as extractor

import torchvision

def remove_final_dir_from_path(path):
    """
    Removes the final directory from the given path string and returns the modified path.
    
    Parameters:
    - path: The original file path (str).
    
    Returns:
    - A string with the final directory component removed from the path.
    """
    # Split the path into components
    path_parts = path.split(os.sep)
    
    # Remove the last directory component
    modified_path_parts = path_parts[:-1]
    
    # Reconstruct the path without the final directory
    modified_path = os.sep.join(modified_path_parts)
    
    return modified_path

class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))
        
        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath

    def automatic_configure(self, models_dirpath: str):
        return True

    def manual_configure(self, models_dirpath: str):
        return True
    
    def infer(self,model_filepath,result_filepath,scratch_dirpath,examples_dirpath,round_training_dataset_dirpath):

        interface = helper.engine(folder=remove_final_dir_from_path(model_filepath))
        p_list = []

        features = [extractor.get_feature_conv(interface)['fvs']]
        if not self.learned_parameters_dirpath is None:
            try:
                ensemble=torch.load(os.path.join(self.learned_parameters_dirpath, f'conv.pt'),map_location=torch.device('cpu'))
            except:
                ensemble=torch.load(os.path.join('/',self.learned_parameters_dirpath, f'conv.pt'),map_location=torch.device('cpu'))
            trojan_probability=extractor.predict(ensemble, features, input_dim=18)
        else:
            trojan_probability=0.5
        p_list.append(trojan_probability)

        # features = [extractor.get_feature(interface)['fvs']]
        # if not self.learned_parameters_dirpath is None:
        #     try:
        #         ensemble=torch.load(os.path.join(self.learned_parameters_dirpath, f'w-6.pt'),map_location=torch.device('cpu'))
        #     except:
        #         ensemble=torch.load(os.path.join('/',self.learned_parameters_dirpath, f'w-6.pt'),map_location=torch.device('cpu'))
        #     trojan_probability=extractor.predict(ensemble, features, input_dim=6)
        # else:
        #     trojan_probability=0.5
        # p_list.append(trojan_probability)


        # features = [extractor.get_feature_out(interface)['fvs']]        
        # if not self.learned_parameters_dirpath is None:
        #     try:
        #         ensemble=torch.load(os.path.join(self.learned_parameters_dirpath, f'm_f5.pt'),map_location=torch.device('cpu'))
        #     except:
        #         ensemble=torch.load(os.path.join('/',self.learned_parameters_dirpath, f'm_f5.pt'),map_location=torch.device('cpu'))
        #     trojan_probability=extractor.predict(ensemble, features, input_dim=2)
        # else:
        #     trojan_probability=0.5
        # p_list.append(trojan_probability)

        trojan_probability = sum(p_list) / len(p_list)
        
        print(trojan_probability)
        
        with open(result_filepath, "w") as fp:
            fp.write('%f'%trojan_probability)
        
        logging.info("Trojan probability: %f", trojan_probability)
        return trojan_probability
    
    
