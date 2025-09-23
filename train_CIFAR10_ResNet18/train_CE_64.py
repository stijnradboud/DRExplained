import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from time import localtime, strftime
import math
from resnet18 import make_resnet18k

# Full CIFAR10 train set
train_dataset = torchvision.datasets.CIFAR10(root='../datasets', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=False)


# Number of optimizer iterations to run, this is the number used in the paper.
# its_end = int(1e6)
# Please be aware also that training a Resnet in 64 bit can be very slow
its_end = int(1e2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

out_file="../logs/output_CIFAR10_CE_64.txt"

with open(out_file, "a") as f:
    f.write(f"Device = {device}\n")

# seed for the model and dataloader shuffling
seed = 42

modeldir = f'../checkpoints/models_resnet/models_CE_64'
os.makedirs(modeldir, exist_ok=True)
torch.manual_seed(seed)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, 
                                            shuffle=True) 

with open(out_file, "a") as f:
    f.write(f"Starting seed {seed} at {strftime('%H:%M:%S', localtime())}\n")

model = make_resnet18k(k=16, num_classes=10, bn=False)

model = model.to(torch.float64).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

its = 0
model.train()

while its < its_end + 1:

    for images, labels in train_loader:
        model.train()
        
        images = images.to(device, dtype=torch.float64)
        
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
            log = its % (10**(math.floor(math.log10(its))-1)) == 0
        else:
            log = True 

        if log:
            torch.save(model, f'{modeldir}/model_at_{its}_its.ckpt')

            with open(out_file, "a") as f:
                f.write(f"\tits: {its} at at {strftime('%H:%M:%S', localtime())}\n")
        
with open(out_file, "a") as f:
    f.write(f"\tDONE with seed {seed} at {strftime('%H:%M:%S', localtime())}")
