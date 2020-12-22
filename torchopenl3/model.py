import warnings
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchopenl3.core import _load_openl3_model, _load_spectrogram_model

class OpenL3Embedding(pl.LightningModule):

    def __init__(self, 
                input_repr: str = 'mel256', 
                embedding_size: int = 512, 
                content_type: str = 'music',
                pretrained: bool = True, 
                use_precomputed_spectrograms: bool = False):
        super().__init__()

        self.use_precomputed_spectrograms = use_precomputed_spectrograms
        if not self.use_precomputed_spectrograms:
            self.filters = _load_spectrogram_model(input_repr)

        self.openl3 = _load_openl3_model(input_repr=input_repr, embedding_size=embedding_size, 
                                        content_type=content_type, pretrained=pretrained)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        if not self.use_precomputed_spectrograms:
            x = self.filters(x)
        x = self.openl3(x)
        return self.flatten(x)