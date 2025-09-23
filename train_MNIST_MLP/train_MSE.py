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
import torch.multiprocessing as mp

def train_network(eps=1e-8, device='cuda:0'):
    
    out_file=f"../logs/output_MNIST_MSE_eps={eps}.txt"
    
    try:    

        train_dataset = torchvision.datasets.MNIST(root='../datasets', 
                                                   train=True, 
                                                   transform=transforms.ToTensor(),  
                                                   download=True)
            
        # Load the 1000 indices
        with open('../datasets/subsample_train_indices.pkl', 'rb') as f:
            subsample_train_indices = pickle.load(f)
        
        # Data loader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, 
                                                   sampler=SubsetRandomSampler(subsample_train_indices)) 
        
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        with open(out_file, "a") as f:
            f.write(f"Device = {device}\n")

        # Seed for the model
        seed = 42
        modeldir = f'../checkpoints/models_MNIST/models_MSE_eps={eps}'

        os.makedirs(modeldir, exist_ok=True)
  
        torch.manual_seed(seed)

        with open(out_file, "a") as f:
            f.write(f"Starting seed {seed} at {strftime('%H:%M:%S', localtime())}\n")
    
        input_size = 784
        hidden_size = 200
        num_classes = 10

        model = nn.Sequential(
                    nn.Linear(input_size, hidden_size), nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                    nn.Linear(hidden_size, num_classes)
                )
        
        model = model.to(device)
        
        criterion = torch.nn.MSELoss()

        if eps == "AMSGrad":
            optimizer = torch.optim.Adam(model.parameters(), amsgrad=True) 
        else:
            optimizer = torch.optim.Adam(model.parameters(), eps=eps) 
        
        its = 0
        model.train()

        # Number of optimizer iterations to run, this is the number used in the paper.
        # Can be reduced massively to still show the effect of AMSGrad on delayed robustness
        its_end = int(1e3)

        torch.save(model, f'{modeldir}/model_at_{its}_its.ckpt')

        while its < its_end + 1:
            for images, labels in train_loader:
                model.train()
                
                images = images.reshape(-1, 28 * 28).to(device)
                labels_one_hot = F.one_hot(labels, num_classes=10).float().to(device)
        
                outputs = model(images)
                
                loss = criterion(outputs, labels_one_hot)
                               
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
                
    except Exception as e:
        with open(out_file, "a") as f:
            f.write(f"Thread failed with error: {e}\n")

if __name__ == '__main__':
    mp.set_start_method('spawn')

    # will download the dataset if it does not exist yet so that does not happen in each subprocess.

    _ = torchvision.datasets.MNIST(root='../datasets', 
                                                   train=True, 
                                                   transform=transforms.ToTensor(),  
                                                   download=True)

    processes = []
    # The three experiments, different Adam epsilons (refered to with delta in the paper) and AMSGrad
    for eps in [1e-4, 1e-8, "AMSGrad"]:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        p = mp.Process(target=train_network, args=(eps, device))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()