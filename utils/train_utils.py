import torch
from time import localtime, strftime
import torch.nn.functional as F
import os
import math
import pickle
from utils.eval_utils import test_model_AAPlus, test_model_PGD

def train_network(model, ID, train_loader, loss_fn="CE", optim="Adam", prec=32, its_end=int(1e3), device=torch.device("cuda:0"), 
                  save_optimizer = False, save_full_losses=True, 
                  attack_PGD=False, attack_AA=False, adv_test_loader=None):

    out_file=f"logs/output_{ID}_{loss_fn}_{optim}_{prec}.txt"
    modeldir = f"checkpoints/models_{ID}/models_{loss_fn}_{optim}_{prec}"
    os.makedirs(modeldir, exist_ok=True)

    if save_optimizer:
        optimdir = f"checkpoints/optimizers_{ID}/optimizers_{loss_fn}_{optim}_{prec}"
        os.makedirs(optimdir, exist_ok=True)

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

    train_losses = []

    PGD_accs = []
    AA_accs = []

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
            if save_full_losses:
                train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if attack_PGD:
                if adv_test_loader is not None:
                    PGD_acc = test_model_PGD(model, adv_test_loader, device)
                    PGD_accs.append(PGD_acc)
                else:
                    print("Could not attack during training as no dataloader with samples to perturb was provided!")
                    attack_PGD = False
                    attack_AA = False
            
            if attack_AA:
                if adv_test_loader is not None:
                    AA_acc = test_model_AAPlus(model_dir=None, i=0,test_loader=adv_test_loader, model=model, device=device) # TODO: fix function so no need for dummy_vars
                    AA_accs.append(AA_acc)
                else:
                    print("Could not attack during training as no dataloader with samples to perturb was provided!")
                    attack_AA = False
                
            its += 1

            # Logarithmic checkpointing
            # [1, .., 10], [11, ..., 99], [100, 110, ..., 990], [1000, 1100, ..., 9900], ...
            if its >= 10:
                log = its % (10**(math.floor(math.log10(its))-1)) == 0
            else:
                log = True 

            if log:
                torch.save(model, f'{modeldir}/model_at_{its}_its.ckpt')

                if save_optimizer:
                    torch.save(optimizer, f'{optimdir}/optim_at_{its}_its.optim')

                with open(out_file, "a") as f:
                    f.write(f"\tits: {its} at at {strftime('%H:%M:%S', localtime())}\n")

                if save_full_losses:
                    with open(f'{modeldir}/losses.pkl', 'wb'):
                        pickle.dump(train_losses, f)
                
                if attack_PGD:
                    with open(f'{modeldir}/PGDaccs.pkl', 'wb'):
                        pickle.dump(PGD_accs, f)
                    
                if attack_AA:
                    with open(f'{modeldir}/AAAccs.pkl', 'wb'):
                        pickle.dump(AA_accs, f)

            if its > its_end:
                break
    
    with open(out_file, "a") as f:
        f.write(f"\tDONE at {strftime('%H:%M:%S', localtime())}")