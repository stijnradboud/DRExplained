import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle
from torch.utils.data import SubsetRandomSampler
import torch.multiprocessing as mp
from utils.train_utils import train_network

# List of models to train in parallel (loss_fn ("CE"/"MSE"), optim ("Adam"/"AMSGrad"), precision (32/64), iterations (int))
experiments = [("CE", "Adam", 32, int(1e6)), ("CE", "Adam", 64, int(1e6)),
               ("MSE", "Adam", 32, int(1e7)), ("MSE", "AMSGrad", 32, int(1e7))]

# If you run out of (cuda) memory, run one at a time individually or change device logic below.

# This is the configuration used in the paper.
# experiments = [("CE", "Adam", 32, int(1e6)), ("CE", "Adam", 64, int(1e6)), ("MSE", "Adam", 32, int(1e7)), ("MSE", "AMSGrad", 32, int(1e7))]

# wether to save optimizers as well as models at checkpoints (default False):
save_optimizers = True

# wether to obtain PGD or AA results at every train it. 
track_PGD = False
track_AA = False

def start_train_loop(loss_fn="CE", optim="Adam", prec=32, its_end=int(1e3), device='cuda:0'):
    try:
        train_dataset = torchvision.datasets.MNIST(root='datasets', 
                                                    train=True, 
                                                    transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Lambda(lambda x: x.view(-1))
                                                    ]),
                                                    download=True)
        # Load the 1000 indices
        with open('datasets/subsample_train_indices.pkl', 'rb') as f:
            subsample_train_indices = pickle.load(f)
            
        # Data loader for training
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, 
                                                    sampler=SubsetRandomSampler(subsample_train_indices))   
        if track_PGD or track_AA: 
            test_dataset = torchvision.datasets.MNIST(root='datasets', 
                                                        train=False, 
                                                        transform=transforms.Compose([
                                                            transforms.ToTensor(),
                                                            transforms.Lambda(lambda x: x.view(-1))
                                                        ]),
                                                        download=True)
            with open('datasets/subsample_test_indices.pkl', 'rb') as f:
                subsample_test_indices = pickle.load(f)
            # Data loader for training
            adv_test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, 
                                                    sampler=SubsetRandomSampler(subsample_test_indices))
        else: 
            adv_test_loader = None
          
        # Seed for the model
        seed = 42
        
        torch.manual_seed(seed)

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

        train_network(model, "MNIST", train_loader, loss_fn, optim, prec, its_end, device, save_optimizer=save_optimizers, attack_PGD=track_PGD, attack_AA=track_AA, adv_test_loader=adv_test_loader)
                    
    except Exception as e:
        with open("error_log.txt", "a") as f:
            f.write(f"Thread failed with error: {e}\n")

if __name__ == '__main__':
    mp.set_start_method('spawn')

    # will download the dataset if it does not exist yet so that does not happen in each subprocess.

    _ = torchvision.datasets.MNIST(root='datasets', 
                                                   train=True,  
                                                   download=True)

    processes = []
    # Running the experiments, change device logic here if necessary
    for loss_fn, optim, prec, its_end in experiments:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        p = mp.Process(target=start_train_loop, args=(loss_fn, optim, prec, its_end, device))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()