import math
import torch
import torch.nn.functional as F
from utils.attacks import PGD, ScaledPGD, PGD_with_restarts, ScaledPGD_with_restarts
from autoattack import AutoAttack
import matplotlib.pyplot as plt
from tqdm import tqdm

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
def test_model_AAPlus(model_dir, i, test_loader, eps=0.06, verbose=False, device=torch.device("cuda:0"), only_APGD = False, model = None):
    
    if model is None:
        model = torch.load(f"{model_dir}/model_at_{i}_its.ckpt", weights_only=False, map_location=device)

    def feed_model(data, model):
        return model(data)
    
    l = [x for (x, _) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (_, y) in test_loader]
    y_test = torch.cat(l, 0)

     
    adversary = AutoAttack(lambda x: feed_model(x, model), norm='Linf', eps=eps, version='plus', verbose=verbose)
    if only_APGD:
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

# function to do simply one pass on a model
def test_model_PGD(model, test_dataloader, device, e=0.06, PGD_steps=10, alpha=0.0156):    

    get_att_PGD = lambda model: PGD(model, e, alpha=alpha, steps=PGD_steps, dmax=1, dmin=0)

    att_PGD = get_att_PGD(model)
    
    correct_adv_PGD = 0
    total_test = 0
    
    model.eval()

    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        adv_images_pgd = att_PGD(images, labels)

        # Adversarial accuracy 32-bit PGD
        with torch.no_grad():
            adv_outputs = model(adv_images_pgd)
            _, predicted = torch.max(adv_outputs.data, 1)
            correct_adv_PGD += (predicted == labels).sum().item()

        total_test += labels.size(0)
        
    return (100 * correct_adv_PGD / total_test)

# function to get metrics
def get_metrics(model_dir, to_log, train_dataloader, test_dataloader, device, e=0.06, loss_fn="CE", prec=32, PGD_steps=10, restarts=1, alpha=0.0156):    
    # default alpha = step size = value from Humayun et al.
    if prec == 32:
        image_dtype = torch.float32
    elif prec == 64:
        image_dtype = torch.float64
    else:
        print("unsupported precision")
        return -1
    
    if restarts > 1:
        get_att_PGD = lambda model: PGD_with_restarts(model, e, alpha=alpha, steps=PGD_steps, dmax=0, dmin=0, dtype=image_dtype)
        get_att_ScaledPGD = lambda model: ScaledPGD_with_restarts(model, e, alpha=alpha, steps=PGD_steps, dmax=0, dmin=0, dtype=image_dtype)
    else:
        get_att_PGD = lambda model: PGD(model, e, alpha=alpha, steps=PGD_steps, dmax=1, dmin=0)
        get_att_ScaledPGD = lambda model: ScaledPGD(model, e, alpha=alpha, steps=PGD_steps, dmax=1, dmin=0)

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
            100 * correct_adv_ScaledPGD / total_test
            ))
    return res_own 

# function to get softmax collapse results
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

# function for the dead neuron analysis
def check_dead_neurons(model, train_loader, prec, device=torch.device('cuda')):
    if prec == 32:
        image_dtype = torch.float32
    elif prec == 64:
        image_dtype = torch.float64
    else:
        print("Unsupported precision!")
        return -1

    # assumes train_loader contains entire training set
    images, _ = next(iter(train_loader))

    inp = images.to(image_dtype).to(device)

    dead_per_layer = []

    dead_total = 0

    model.eval()

    for i, layer in enumerate(model):
        with torch.no_grad():
            output = layer(inp)
        if isinstance(layer, torch.nn.Linear):
            print(layer)
            dead_neurons = (output <= 0).all(dim=0)
            num_dead = dead_neurons.sum().item()
            dead_per_layer.append(num_dead)
            dead_total += dead_per_layer
        inp = output
        
    return dead_per_layer, dead_total

def plot_metrics(metrics, checkpoints, loss_fn, file_name, AA_accs = None):
    train_acc = [r[0] for r in metrics]
    test_acc = [r[1] for r in metrics]
    train_loss = [r[2] for r in metrics]
    # test_loss = [r[3] for r in metrics]
    adv_acc_PGD = [r[4] for r in metrics]
    adv_acc_ScaledPGD = [r[5] for r in metrics]

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    axs.plot(checkpoints, train_acc, color='blue', label='Train Accuracy')
    axs.plot(checkpoints, test_acc, color='orange', label='Test Accuracy')
    axs.plot(checkpoints, adv_acc_PGD, color='red', label='Adv. Accuracy (PGD, ε = 0.06)')
    if loss_fn == "CE": 
        axs.plot(checkpoints, adv_acc_ScaledPGD, color='purple', label='Adv. Accuracy (ScaledPGD, ε = 0.06)')
    if AA_accs is not None:
        axs.plot(checkpoints, AA_accs, color='purple', label='Adv. Accuracy (AA+, ε = 0.06)')
    axs.set_xscale("log")
    axs.set_xlabel("Optimizer Iterations")
    axs.set_ylabel("Accuracy (%)")
    axs.grid()
    axs.set_ylim(0, 105)

    ax12 = axs.twinx()

    ax12.plot(checkpoints, train_loss, color='black', label='Train Loss', linestyle='--')
    ax12.set_yscale("log")
    ax12.set_ylabel("Loss")            
    ax12.set_ylim(1e-16, 1)

    handles1, labels1 = axs.get_legend_handles_labels() 
    handles2, labels2= ax12.get_legend_handles_labels()

    handles = handles1 + handles2
    labels = labels1 + labels2
    fig.legend(handles, labels, loc='lower center', ncols=6, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()

    plt.savefig(f"plots/General Metrics - {file_name}.png", dpi=300, bbox_inches='tight')
    plt.close()