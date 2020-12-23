import os

import numpy as np
import torchopenl3

####################
# UTILITIES
####################
def _get_weight_path(content_type, input_repr):
    return os.path.join(torchopenl3.ASSETS_DIR, 'weights', f'{content_type}-{input_repr}')

def _load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

####################
# ASSERTIONS
####################
def _check_embedding_size(embedding_size):
    assert isinstance(embedding_size, int), f'embedding_size must be type int'
    if embedding_size == 512:
        maxpool_kernel=(16, 24)
    elif embedding_size == 6144:
        maxpool_kernel=(4, 8)
    else: 
        raise ValueError(f'embedding_size should be 512 or 6144 but got {embedding_size}')

def _check_content_type(content_type):
     # check content types
    assert isinstance(content_type, str)
    assert content_type in torchopenl3.CONTENT_TYPES, f'content_type must be one of {torchopenl3.CONTENT_TYPES}'

def _check_input_repr(input_repr):
        # check input representation
    assert isinstance(input_repr, str)
    assert input_repr in torchopenl3.INPUT_REPRESENTATIONS,  f'input_repr must be one of {torchopenl3.INPUT_REPRESENTATIONS}'

