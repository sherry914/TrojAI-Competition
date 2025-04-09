from typing import OrderedDict, Dict
import torch

class TrojAIMitigatedModel:
    """This class wraps the output of a mitigation technique in a stardard form. We account for querying the model at one timepoint per prediction required.
    
    If a mitigation technique does not do any data pre or post processing at test time, but just changes the model weights, simply wrap your new state dict in this class:
        new_model = TrojAIMitigatedModel(new_state_dict)

    If a mitigation technique uses a pre/post process transform at test time, you can additionally pass either/or of those functions in and the defaults will be overwritten.
    The call signature of your pre/post process must match the signature of the base version exactly.

        def my_custom_preprocess_fn(x):
            ...
        def my_custom_postprocess_fn(logits, info):
            ...

        new_model = TrojAIMitigatedModel(
            new_state_dict, 
            custom_preprocess=my_custom_preprocess_fn,
            custom_postprocess=my_custom_postprocess_fn
        )

    """
    
    def __init__(self, model, custom_preprocess: callable=None, custom_postprocess: callable=None):
        """
        Args
            state_dict: A state dictionary that can be loaded by the original model
            custom_preprocess: An optional data preprocess function with the same interface as the default one implemented in the class.
            custom_postprocess: An optional data postprocess function with the same interface as the default one implemented in the class.
        """
        self.model = model



    
    