# torchopenl3

pytorch port of the openl3 audio embedding (from the [marl](https://github.com/marl/openl3) implementation)

### Installation

Clone this repo and run `cd torchopenl3 && pip install -e .`

### Usage
OpenL3 comes in a couple of flavors. We can choose from:

- **input representations**: `mel128` or `mel256`. `linear` coming soon
- **content types**: `music` or `env`. The `music` model variant was trained on music, while the `env` was trained on environmental sounds. 
- **embedding size**: output embedding size. Either 512 or 6144. 

Let's load a model! We choose the `mel128`, `music`, `512` variant.

```python 
import torchopenl3
import torch
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if we can!

# dummy audio
SAMPLE_RATE = 48000
audio = np.random.randn(1, SAMPLE_RATE) # 1 second of audio at 48kHz

model = torchopenl3.OpenL3Embedding(input_repr='mel128', 
                                    embedding_size=512, 
                                    content_type='music')

embedding = torchopenl3.embed(model=model, 
                            audio=audio, # shape sould be (channels, samples)
                            sample_rate=SAMPLE_RATE, # sample rate of input file
                            hop_size=1, 
                            device=DEVICE) # use gpu?
```

### Tests

Tests are written using `pytest`. Run `pip install pytest` to install pytest.
run `pytest`.