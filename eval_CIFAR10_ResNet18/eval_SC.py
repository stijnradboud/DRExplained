import math
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
'''
This file creates softmax collapse plots for all the mnist mlp cross entropy experiments.

This file assumes all the mnist mlp train scripts were run as provided.

If this is not the case:
    update the tuple list below
        (name, iterations, precision)
    possibly also update the checkpointing scheme

Note that this can take quite a long time to run, especially for 64-bit networks.
'''

experiments = [("CE_32", int(1e2), 32), ("CE_64", int(1e2), 64)]

# Use only 2000 samples to speed things up. Increase where necessary.
N_SAMPLES_TRAIN = 2000 # max 50k
N_SAMPLES_TEST = 2000 # max 10k

# checkpointing scheme (logarithmic, see train scripts)
def get_checkpoints(its_end):
    checkpoints = []
    for its in range(1, its_end+1):
        if its >= 10:
            log = its % (10**(math.floor(math.log10(its))-1)) == 0
        else:
            log = True   
        if log:
            checkpoints.append(its)
    return checkpoints 

# helper function
def get_SC_results(model, train_loader, prec=32, device=torch.device("cuda")):
    if prec == 32:
        image_dtype = torch.float32
    elif prec == 64:
        image_dtype = torch.float64
    else:
        print("Unsupported precision!")
        return -1
    
    criterion = torch.nn.CrossEntropyLoss()

    n_underflow = 0
    n_absorp = 0
    n_all = 0

    total_train = 0

    for images_train, labels_train in train_loader:
        images_train = images_train.to(image_dtype).to(device)
        labels_train = labels_train.to(device)
                
        outputs = model(images_train)
        
        loss = criterion(outputs, labels_train)

        grad = torch.autograd.grad(
            loss, outputs, retain_graph=False, create_graph=False
        )[0]

        sign = grad.sign()
        
        n_grads = sign.numel()
        
        zeroes = n_grads - torch.count_nonzero(sign).item()
        
        if zeroes > 0:
            for s in range(labels_train.size(0)):
                brokens = []
                for n in range(10):
                    if sign[s, n] == 0:
                        brokens.append(n)
                
                if len(brokens) == 0:
                    continue
                    
                if labels_train[s] in brokens:
                    if len(brokens) == 10:
                        n_all += 1
                        n_absorp += 1
                        n_underflow += 1
                    elif len(brokens) > 1:
                        n_absorp += 1
                        n_underflow += 1
                    else:
                        n_absorp += 1
                else:
                    n_underflow += 1
                
        total_train += labels_train.size(0)

    return (100 * n_underflow / total_train,
            100 * n_absorp / total_train,
            100 * n_all / total_train)

train_dataset = torchvision.datasets.CIFAR10(root='../datasets', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

train_loader = DataLoader(
    train_dataset, 
    batch_size=200, 
    sampler=RandomSampler(train_dataset, num_samples=N_SAMPLES_TRAIN, replacement=False)
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for exp, its_end, prec in tqdm(experiments):
    underflows = []
    absorps = []
    all0 = []

    checkpoints = get_checkpoints(its_end)

    for its in tqdm(checkpoints):
        model = torch.load(f"../checkpoints/models_resnet/models_{exp}/model_at_{its}_its.ckpt", weights_only=False, map_location=device)
        n_underflow, n_absorp, n_all = get_SC_results(model, train_loader, prec, device)
        underflows.append(n_underflow)
        absorps.append(n_absorp)
        all0.append(n_all)
    
    plt.plot(checkpoints, underflows, label='Only Underflow Soft-Max Collapse errors', color='blue')
    plt.plot(checkpoints, absorps, label='Only Absorpion Soft-Max Collapse errors', color='orange')
    plt.plot(checkpoints, all0, label='Gradient completly zero', color='red')

    plt.grid()
    plt.xscale("log")
    
    plt.xlabel("Optimizer iterations")
    plt.ylabel("%")

    plt.ylim(-5, 105)

    plt.legend()

    plt.savefig(f"../plots/Softmax Collapse - CIFAR10 ResNet18 - {exp}.png")

    plt.close()