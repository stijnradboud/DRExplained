# Is Delayed Robustness Really Grokking?
## Authors anonymized until publication.

This is the repository for the paper "Is Delayed Robustness Really Grokking?".

The main purpose of this repo is to provide code to reproduce our results.

Run train_MNIST_MLPs.py and/or train_CIFAR10_ResNet18s.py to train networks. 
See the scripts themselves for configuration.

These will store networks in the 'checkpoints' subfolder(s). You can track their progress using the files in the 'logs' subdirectory.

These stored networks can be evaluated using 'eval_MNIST_MLPs.py' and 'eval_CIFAR10_ResNet18s.py' scripts.
See these scripts themselves for configuration.

They produce plots that go into the 'plots' subfolder.

Reqs (version used):
* python (v3.11.9)
* pythorch (including torchvision etc.) (v2.5.1)
* Matplotlib (3.9.2)
* Numpy (1.24.3)
* tqdm (4.67.1)
* Auto-Attack (see: https://github.com/fra31/auto-attack, 'conda list' returns v0.1).
