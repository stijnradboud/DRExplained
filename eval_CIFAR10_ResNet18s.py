from utils.eval_utils import get_checkpoints, get_metrics, get_SC_results, plot_metrics, test_model_AAPlus
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler
'''
This file creates all the plots in the paper for all the CIFAR10 ResNet18 experiments.

Below is the same list of experiments as in the train file for CIFAR10 ResNet18s.

Update accordingly based on the experiments that were ran.
'''
experiments = [("CE", "Adam", 32, int(1e3)), # ("CE", "Adam", 64, int(1e2)),
               ("MSE", "Adam", 32, int(1e3)), ("MSE", "AMSGrad", 32, int(1e3))]

# Again, if you run out of (cuda) memory, run one at a time individually or change device logic below.

# This is the configuration used in the paper.
# experiments = [("CE", "Adam", 32, int(1e6)), ("CE", "Adam", 64, int(1e6)), ("MSE", "Adam", 32, int(1e7)), ("MSE", "AMSGrad", 32, int(1e7))]

# You can turn on individual analyses/plots below:
analyse_general_metrics = True
analyse_softmax_collapse = True

# Use only 1000 samples to speed things up. Increase where necessary. In the paper we used the full sets.
N_SAMPLES_TRAIN = 256 # max 50k
N_SAMPLES_TEST = 256 # max 10k

# Be warned that evaluating the 64 bit resnet can still take a very long time!

def evaluate_and_plot_checkpoints(loss_fn = "CE", optim="Adam", prec=32, its_end=int(1e3), device=torch.device("cuda:0")):
 
    if prec == 32:
        image_dtype = torch.float32
    elif prec == 64:
        image_dtype = torch.float64
    else:
        print("Unsupported precision!")
        return -1

    train_dataset = torchvision.datasets.CIFAR10(root='datasets', 
                                           train=True, 
                                           transform=transforms.ToTensor(), 
                                           download=False)

    test_dataset = torchvision.datasets.CIFAR10(root='datasets', 
                                            train=False, 
                                            transform=transforms.ToTensor(), 
                                            download=False)


    train_loader = DataLoader(
        train_dataset, 
        batch_size=256, 
        sampler=RandomSampler(train_dataset, num_samples=N_SAMPLES_TRAIN, replacement=False)
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=256, 
        sampler=RandomSampler(test_dataset, num_samples=N_SAMPLES_TEST, replacement=False)
    )

    
    ID = f"{loss_fn}_{optim}_{prec}"

    model_dir = f"checkpoints/models_CIFAR10/models_{ID}"

    print(f"Evaluating {ID}...")
    checkpoints = get_checkpoints(its_end)

    if analyse_general_metrics:
        print("\tAccuracies, losses and adversarial accuracies...")
        metrics = get_metrics(model_dir, checkpoints, train_loader, test_loader, device, loss_fn=loss_fn, prec=prec)

        if loss_fn == "MSE":
            print("\tAA+ attack...")
            AA_accs = []
            for its in tqdm(checkpoints):
                aa_acc = test_model_AAPlus(model_dir, its, test_loader, device=device)
                AA_accs.append(aa_acc)
        else:
            AA_accs = None

        plot_metrics(metrics, checkpoints, loss_fn, f"CIFAR10 ResNet18 - {ID}", APGD_Accs=AA_accs)
            

    if analyse_softmax_collapse and loss_fn == "CE":
        print(f"\tSC...")
        underflows = []
        absorps = []
        all0 = []
        for its in tqdm(checkpoints):
            model = torch.load(f"{model_dir}/model_at_{its}_its.ckpt", weights_only=False, map_location=device)
            model.eval()
            if analyse_softmax_collapse and loss_fn == "CE":
                n_underflow, n_absorp, n_all = get_SC_results(model, train_loader, prec, device)
                underflows.append(n_underflow)
                absorps.append(n_absorp)
                all0.append(n_all)

        if analyse_softmax_collapse and loss_fn == "CE":
            plt.plot(checkpoints, underflows, label='Only Underflow Soft-Max Collapse errors', color='blue')
            plt.plot(checkpoints, absorps, label='Only Absorpion Soft-Max Collapse errors', color='orange')
            plt.plot(checkpoints, all0, label='Gradient completly zero', color='red')
            plt.grid()
            plt.xscale("log")
            plt.xlabel("Optimizer iterations")
            plt.ylabel("%")
            plt.ylim(-5, 105)
            plt.legend()
            plt.savefig(f"plots/Softmax Collapse - CIFAR10 ResNet18 - {ID}.png")
            plt.close()

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for loss_fn, optim, prec, its_end in experiments:
        evaluate_and_plot_checkpoints(loss_fn, optim, prec, its_end)