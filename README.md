# Deep learning project template

Throughout this template, `NAME` is used to refer to the name of the project
and `DATASET` is used to refer to the name of a dataset.


### Installation

Clone this repo and run `cd NAME && pip install -e .`.

### Usage

##### Download data

Place datasets in `data/DATASET`, where `DATASET` is the name of the dataset.


##### Partition data

Complete all TODOs in `partition.py`, then run `python -m NAME.partition
DATASET`.


##### Preprocess data

Complete all TODOs in `preprocess.py`, then run `python -m NAME.preprocess
DATASET`. All preprocessed data is saved in `cache/DATASET`.


##### Train

Complete all TODOs in `data.py` and `model.py`. Then, create a directory in
`runs` for your experiment. Logs, checkpoints, and results should be saved to
this directory. In your new directory, run `python -m NAME.train --dataset
DATASET <args>`. See the [PyTorch Lightning trainer flags](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#trainer-flags)
for additional arguments.


##### Infer

Complete all TODOs in `infer.py`, then run `python -m NAME.infer
<input_file> <output_file> <checkpoint_file>`.


##### Monitor

Run `tensorboard --logdir runs/<run>/logs`. If you are running training
remotely, you must create a SSH connection with port forwarding to view
Tensorboard. This can be done with `ssh -L 6006:localhost:6006
<user>@<server-ip-address>`. Then, open `localhost:6006` in your browser.


##### Test

Tests are written using `pytest`. Run `pip install pytest` to install pytest.
Complete all TODOs in `test_model.py` and `test_data.py`, then run `pytest`.
Adding project-specific tests for preprocessing and inference is encouraged.
