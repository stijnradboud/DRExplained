import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import pickle
from torch.utils.data import SubsetRandomSampler
from time import localtime, strftime
import math

# MNIST dataset, 1K samples
train_dataset = torchvision.datasets.MNIST(root='data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

# Load the indices of the 1k susbet
with open('subsample_train_indices.pkl', 'rb') as f:
    subsample_train_indices = pickle.load(f)

# Data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, 
                                           sampler=SubsetRandomSampler(subsample_train_indices)) 

# Number of optimizer iterations to run, this is the number used in the paper.
its_end = int(1e6)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

out_file="../logs/output_MNIST_CE_32.txt"

with open(out_file, "a") as f:
    f.write(f"Device = {device}\n")

# Seed for the model and dataloader shuffling
seed = 42

modeldir = f'../checkpoints/models_MNIST/models_CE_32'
os.makedirs(modeldir, exist_ok=True)
torch.manual_seed(seed)

with open(out_file, "a") as f:
    f.write(f"Starting seed {seed} at {strftime('%H:%M:%S', localtime())}\n")

input_size = 784
hidden_size = 200
num_classes = 10

model = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), 
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, num_classes)
                )

model = model.to(torch.float64).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

its = 0
model.train()

while its < its_end + 1:
    for images, labels in train_loader:
        model.train()
        
        images = images.reshape(-1, 28 * 28).to(device, dtype=torch.float64)
        
        labels = labels.to(device, dtype=torch.int64)

        outputs = model(images)
        loss = criterion(outputs, labels) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        its += 1

        # Logarithmic checkpointing
        # [1, .., 10], [11, ..., 99], [100, 110, ..., 990], [1000, 1100, ..., 9900], ...
        if its >= 10:
            log = its % (10**(math.floor(math.log10(its)))-1) == 0
        else:
            log = True

        if log:
            torch.save(model, f'{modeldir}/model_at_{its}_its.ckpt')

            with open(out_file, "a") as f:
                f.write(f"\tits: {its} at at {strftime('%H:%M:%S', localtime())}\n")
        
with open(out_file, "a") as f:
    f.write(f"\tDONE with seed {seed} at {strftime('%H:%M:%S', localtime())}")
