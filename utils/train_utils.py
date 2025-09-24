import torch
from time import localtime, strftime
import torch.nn.functional as F
import os
import math

def train_network(model, ID, train_loader, loss_fn="CE", optim="Adam", prec=32, its_end=int(1e3), device=torch.device("cuda:0")):

    out_file=f"logs/output_{ID}_{loss_fn}_{optim}_{prec}.txt"
    modeldir = f"checkpoints/models_{ID}/models_{loss_fn}_{optim}_{prec}"

    os.makedirs(modeldir, exist_ok=True)

    with open(out_file, "w") as f:
            f.write(f"Starting at {strftime('%H:%M:%S', localtime())}\n")

    if loss_fn == "CE":
        criterion = torch.nn.CrossEntropyLoss() 
    elif loss_fn == "MSE":
        criterion = torch.nn.MSELoss()
    else:
        print("Unsupported loss function!")
        return -1

    if optim == "AMSGrad":
        optimizer = torch.optim.Adam(model.parameters(), amsgrad=True) 
    elif optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters())
    else:
        print("Unsupported optimizer!")
        return -1

    if prec == 32:
        image_dtype = torch.float32
    elif prec == 64:
        image_dtype = torch.float64
    else:
        print("Unsupported precision!")
        return -1
    
    its = 0

    model = model.to(image_dtype).to(device)

    model.train()

    while its < its_end + 1:
        for images, labels in train_loader:
    
            images = images.to(device, dtype=image_dtype)
            
            labels = labels.to(device)

            outputs = model(images)

            if loss_fn == "MSE":
                labels_one_hot = F.one_hot(labels, num_classes=10).to(image_dtype).to(device)
                loss = criterion(outputs, labels_one_hot)
            else:
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

            if its > its_end:
                break
    
    with open(out_file, "a") as f:
        f.write(f"\tDONE at {strftime('%H:%M:%S', localtime())}")