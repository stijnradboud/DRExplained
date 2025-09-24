import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from torch.utils.data import SubsetRandomSampler
import math
import matplotlib.pyplot as plt

'''
This file creates dead neuron plots for all the mnist mlp experiments.

This file assumes all the mnist mlp train scripts were run as provided.

If this is not the case:
    update the tuple list below
    possibly also update the checkpointing scheme
'''

experiments = [(f"MSE_eps={1e-8}", int(1e3)), ("MSE_eps=AMSGrad", int(1e3)), ("CE_32", int(1e3), ("CE_64", int(1e3)))]

def get_checkpoints(its_end):
    checkpoints = []
    for its in range(1, its_end+1):
        if its >= 10:
            log = its % (10**(math.floor(math.log10(its)))-1) == 0
        else:
            log = True
        if log:
            checkpoints.append(its)
    return checkpoints

# Helper function
def check_dead_neurons_per_layer(model, train_loader, device=torch.device('cuda')):
    # assumes train_loader contains entire training set
    images, _ = next(iter(train_loader))

    inp = images.reshape(-1, 28*28).to(device)

    dead_per_layer = []

    dead_total = 0

    model.eval()
    for i, layer in enumerate(model):
        with torch.no_grad():
            output = layer(inp)
        if i % 2:
            dead_neurons = (output == 0).all(dim=0)
            num_dead = dead_neurons.sum().item()
            dead_per_layer.append(num_dead)
            dead_total += num_dead
        inp = output
        
    return dead_per_layer, dead_total

# dataloader
train_dataset = torchvision.datasets.MNIST(root='../datasets', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=False)
with open('../datasets/subsample_train_indices.pkl', 'rb') as f:
    subsample_train_indices = pickle.load(f)

# Please put the entire 1k dataset into a batch!
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, 
                                           sampler=SubsetRandomSampler(subsample_train_indices), shuffle=False) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Calculate dead neurons
results = []
for exp, its_end in experiments:
    checkpoints = get_checkpoints(its_end)
    dead_neurons = []
    for i in checkpoints:
        model = torch.load(f"../checkpoints/models_MIST/models_{exp}/model_at_{i}_its.ckpt", weights_only = False, map_location=torch.device('cuda'))
    
        _, sum = check_dead_neurons_per_layer(model, train_loader)
        dead_neurons.append(sum)
    results.append((exp, checkpoints, dead_neurons))

plt.figure(figsize=(10,3))

for exp, checkpoints, dead_neurons in results:
    plt.plot(checkpoints, dead_neurons, label = exp)

plt.legend()
plt.xscale("log")
plt.xlabel("Optimizer Iterations")
plt.ylabel("% Dead Neurons across Network")
plt.ylim(0, 100)

plt.savefig("../plots/Dead Neurons - MNIST MLPs.pdf", dpi=300, bbox_inches='tight')

plt.close()