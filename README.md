# Is Delayed Robustness Really Grokking?
## Authors anonymized until publication.

This is the repository for the paper "Is Delayed Robustness Really Grokking?".

In 'train_MNIST_MLP' and 'train_CIFAR10_ResNet18' one can find code to train networks like ours.

This will store networks in the 'checkpoints' subfolder(s).

These can be evaluated using the scripts in 'eval_MNIST_MLP' and 'eval_CIFAR10_ResNet18'.

They produce plots that go into the 'plots' subfolder.

The plots should be reproductions of our main figures.

Reqs:
* python (v3.11.9)
* pythorch (including torchvision etc.) (v2.5.1)
* Matplotlib (3.9.2)
* Numpy (1.24.3)
* Autoattack (see: https://github.com/fra31/auto-attack).
