"""core.py - root module exports"""

from pathlib import Path

import numpy as np
import torch

from .model import OpenL3Embedding

###############################################################################
# Constants
###############################################################################

#  Static directories
ASSETS_DIR = Path(__file__).parent / 'assets'
CACHE_DIR = Path(__file__).parent.parent / 'cache'
DATA_DIR = Path(__file__).parent.parent / 'data'

#  Model Parameters
SAMPLE_RATE = 48000
INPUT_REPRESENTATIONS = ('mel128', 'mel256')
CONTENT_TYPES = ('music', 'env')
EMBEDDING_SIZES = (512, 6144)
POOLING_SIZES = {
    'linear': {
        6144: (8, 8),
        512: (32, 24),
    },
    'mel128': {
        6144: (4, 8),
        512: (16, 24),
    },
    'mel256': {
        6144: (8, 8),
        512: (32, 24),
    }
}

def all_models():
    import itertools
    for input_repr, embedding_size, content_type in itertools.product(INPUT_REPRESENTATIONS, 
                                                                      EMBEDDING_SIZES, 
                                                                      CONTENT_TYPES):
        model = OpenL3Embedding(input_repr=input_repr, embedding_size=embedding_size, content_type=content_type)
        yield model