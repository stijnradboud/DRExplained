from attacks import PGD, ScaledPGD
import math
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from autoattack import AutoAttack
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
'''
This file creates loss/accuracy plots for all the mnist mlp experiments.

It plots: train set loss, train set acc, test set acc 
and adversarial accuracy to pgd and scaledpgd (at eps=0.06 perturbation) where applicable

This file assumes all the mnist mlp train scripts were run as provided.

If this is not the case:
    update the tuple list below
    (name, iterations, loss_function, precision)
    possibly also update the checkpointing scheme
'''

experiments = [(f"MSE_Adam", int(1e2), "MSE", 32), ("MSE_AMSGrad", int(1e2), "MSE", 32),
               ("CE_32", int(1e2), "CE", 32), ("CE_64", int(1e2), "CE", 64)]

# Use only 1000 samples to speed things up. Increase where necessary. In the paper we used the full sets.
N_SAMPLES_TRAIN = 1000 # max 50k
N_SAMPLES_TEST = 1000 # max 10k

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

# function to use auto attack
def test_model_APGD(model, test_loader, eps=0.06, verbose=False):

    def feed_model(data, model):
        return model(data)
    
    l = [x for (x, _) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (_, y) in test_loader]
    y_test = torch.cat(l, 0)

     
    adversary = AutoAttack(lambda x: feed_model(x, model), norm='Linf', eps=eps, version='plus', verbose=verbose)  
    adversary.attacks_to_run = ['apgd-ce']
    adversary.seed = 42
            
    with torch.no_grad():
        adv_complete = adversary.run_standard_evaluation(x_test, y_test,
                    bs=200)

    flattened_images = adv_complete.to(device)
    model.eval()
    with torch.no_grad():
        # Get model predictions (logits)
        outputs = model(flattened_images)
    
    # Get predicted labels (taking the index of the max logit per sample)
    _, predicted = torch.max(outputs, 1)
    
    # Calculate accuracy
    correct = (predicted == y_test.to(device))
    total = y_test.size(0)
    accuracy = correct.sum().item() / total * 100    

    return accuracy

# function to get metrics
def get_metrics(model_dir, to_log, train_dataloader, test_dataloader, device, e=0.06, loss_fn="CE", prec=32):    
    
    if prec == 32:
        image_dtype = torch.float32
    elif prec == 64:
        image_dtype = torch.float64
    else:
        print("unsupported precision")
        return -1
    
    get_att_PGD = lambda model: PGD(model, e, alpha=0.0156, steps=10, dmax=1, dmin=0)
    get_att_ScaledPGD = lambda model: ScaledPGD(model, e, alpha=0.0156, steps=10, dmax=1, dmin=0)

    if loss_fn == "CE":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_fn == "MSE":
        criterion = torch.nn.MSELoss()
    else:
        print("unsupported loss function")
        return -1
    
    res_own = []

    for its in tqdm(to_log):
        model = torch.load(f"{model_dir}/model_at_{its}_its.ckpt", weights_only=False, map_location=device)

        att_PGD = get_att_PGD(model)
        att_ScaledPGD = get_att_ScaledPGD(model)
        
        correct_clean_train = 0
        correct_clean_test = 0
        correct_adv_PGD = 0
        correct_adv_ScaledPGD = 0
        
        total_train_loss = 0
        total_test_loss = 0
        
        total_train = 0
        total_test = 0
        
        model.eval()

        if loss_fn == "MSE":
            acc_APGD = test_model_APGD(model, test_dataloader)
        else:
            acc_APGD = 0

        for images, labels in train_dataloader:
            images = images.to(image_dtype).to(device)
            labels = labels.to(device)

            if loss_fn == "MSE":
                inp_labels = F.one_hot(labels, num_classes=10).to(image_dtype).to(device)
            else:
                inp_labels = labels                


            # Clean accuracy
            with torch.no_grad():
                outputs = model(images).to(image_dtype)
                     
                loss = criterion(outputs, inp_labels)
    
                total_train_loss += loss.item()
               
                _, predicted = torch.max(outputs.data, 1)
                correct_clean_train += (predicted == labels).sum().item()
                    
            total_train += labels.size(0)
        
        
        for images, labels in test_dataloader:
            images = images.to(image_dtype).to(device)
            labels = labels.to(device)

            if loss_fn == "MSE":
                inp_labels = F.one_hot(labels, num_classes=10).to(image_dtype).to(device)
            else:
                inp_labels = labels                

            adv_images_pgd = att_PGD(images, labels)

            adv_images_scale = att_ScaledPGD(images, labels)
            
            # Clean accuracy
            with torch.no_grad():
                outputs = model(images)
                     
                loss = criterion(outputs, inp_labels)
    
                total_test_loss += loss.item()
               
                _, predicted = torch.max(outputs.data, 1)
                correct_clean_test += (predicted == labels).sum().item()
                
            # Adversarial accuracy 32-bit PGD
            with torch.no_grad():
                adv_outputs = model(adv_images_pgd)
                _, predicted = torch.max(adv_outputs.data, 1)
                correct_adv_PGD += (predicted == labels).sum().item()

            if loss_fn == "CE":
                # Adversarial accuracy scaled PGD
                with torch.no_grad():
                    adv_outputs_scale = model(adv_images_scale)
                    _, predicted = torch.max(adv_outputs_scale.data, 1)
                    correct_adv_ScaledPGD += (predicted == labels).sum().item()
            
            total_test += labels.size(0)
        
        res_own.append((
            100 * correct_clean_train / total_train,
            100 * correct_clean_test /total_test,
            total_train_loss / total_train,
            total_test_loss / total_test,
            100 * correct_adv_PGD / total_test,
            100 * correct_adv_ScaledPGD / total_test,
            acc_APGD
            ))
    return res_own 

train_dataset = torchvision.datasets.CIFAR10(root='../datasets', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../datasets', 
                                          train=False, 
                                          transform=transforms.ToTensor(),
                                          download=True)


train_loader = DataLoader(
    train_dataset, 
    batch_size=200, 
    sampler=RandomSampler(train_dataset, num_samples=N_SAMPLES_TRAIN, replacement=False)
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=200, 
    sampler=RandomSampler(test_dataset, num_samples=N_SAMPLES_TEST, replacement=False)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for exp, its, loss_fn, prec in tqdm(experiments):
    checkpoints = get_checkpoints(its)
    model_dir = f"../checkpoints/models_resnet/models_{exp}"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    metrics = get_metrics(model_dir, checkpoints, train_loader, test_loader, device, loss_fn=loss_fn, prec=prec)


    train_acc = [r[0] for r in metrics]
    test_acc = [r[1] for r in metrics]
    train_loss = [r[2] for r in metrics]
    # test_loss = [r[3] for r in metrics]
    adv_acc_PGD = [r[4] for r in metrics]
    adv_acc_ScaledPGD = [r[5] for r in metrics]
    adv_acc_APGD = [r[6] for r in metrics]

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    axs.plot(checkpoints, train_acc, color='blue', label='Train Accuracy')
    axs.plot(checkpoints, test_acc, color='orange', label='Test Accuracy')
    axs.plot(checkpoints, adv_acc_PGD, color='red', label='Adv. Accuracy (PGD, ε = 0.06)')
    if loss_fn == "CE": 
        axs.plot(checkpoints, adv_acc_ScaledPGD, color='purple', label='Adv. Accuracy (ScaledPGD, , ε = 0.06)')
    if loss_fn == "MSE":
        axs.plot(checkpoints, adv_acc_APGD, color='purple', label='Adv. Accuracy (Auto-PGD, , ε = 0.06)')
    axs.set_xscale("log")
    axs.set_xlabel("Optimizer Iterations")
    axs.set_ylabel("Accuracy (%)")
    axs.grid()
    axs.set_ylim(0, 105)

    ax12 = axs.twinx()

    ax12.plot(checkpoints, train_loss, color='black', label='Train Loss', linestyle='--')
    ax12.set_yscale("log")
    ax12.set_ylabel("Loss")            
    ax12.set_ylim(1e-10, 1)

    handles1, labels1 = axs.get_legend_handles_labels() 
    handles2, labels2= ax12.get_legend_handles_labels()

    handles = handles1 + handles2
    labels = labels1 + labels2
    fig.legend(handles, labels, loc='lower center', ncols=6, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()

    plt.savefig(f"../plots/Accuracy - CIFAR10 ResNet18 - {exp}.png", dpi=300, bbox_inches='tight')
    plt.close()