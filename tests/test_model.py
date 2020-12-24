import pytest
import numpy as np
import torch
import torchopenl3

###############################################################################
# Test model
###############################################################################

@pytest.mark.parametrize("model", [m for m in torchopenl3.all_models()])
def test_model_output_shape(model):
    audio = np.random.rand(1, 48000)
    audio = torch.from_numpy(audio).unsqueeze(0).float()

    expected_shape = (1, model.embedding_size)
    print(model.content_type)

    assert model(audio).shape == expected_shape