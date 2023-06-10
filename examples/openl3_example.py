#!/usr/bin/env

"""
Sample torchopenl3 script that supports alternate installation method. 
Instead of cloning repo and running "pip install -e ." ,
this verision supports install via "pip install git+https://github.com/hugofloresgarcia/torchopenl3.git", which would not normally
install the weights, so this program grabs the weights and installs them where they need to go
"""


print("importing")
import torchopenl3
import torch
import numpy as np
import os
from tqdm import tqdm 
import requests
from aeiou.core import makedir
import site 
print("imports finished")

def download_file(url, local_filename):
    "Includes a progress bar.  from https://stackoverflow.com/a/37573701/4259243"
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kilobye
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(local_filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    return local_filename


def download_if_needed(url, local_filename):
    "wrapper for download_file"
    if not os.path.isfile(local_filename):
        print(f"File {local_filename} not found, downloading from {url}")
        download_file( url, local_filename)
    return local_filename



print("checking on model weights")
# download and install weights first
weights_dir = f"{site.getsitepackages()[0]}/torchopenl3/assets/weights"
makedir(weights_dir)
input_repr='mel128'
weights_file = f"env-{input_repr}"
download_if_needed(f"https://github.com/hugofloresgarcia/torchopenl3/raw/main/torchopenl3/assets/weights/{weights_file}", f"{weights_dir}/{weights_file}")



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if we can!


# dummy audio
SAMPLE_RATE = 48000
audio = np.random.randn(1, SAMPLE_RATE).astype(np.float32) # 1 second of audio at 48kHz
print("dtype is ",audio.dtype)


print("setting up model")
model = torchopenl3.OpenL3Embedding(input_repr=input_repr, 
                                    embedding_size=512, 
                                    content_type='music')

print("doing embedding")
embedding = torchopenl3.embed(model=model, 
                            audio=audio, # shape sould be (channels, samples)
                            sample_rate=SAMPLE_RATE, # sample rate of input file
                            hop_size=1, 
                            device=DEVICE) # use gpu?

print("embedding.shape =",embedding.shape)
print("finished")
