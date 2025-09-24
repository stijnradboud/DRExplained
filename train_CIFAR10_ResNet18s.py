import torch
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from utils.train_utils import train_network
from utils.resnet18 import make_resnet18k

# List of models to train in parallel (loss_fn ("CE"/"MSE"), optim ("Adam"/"AMSGrad"), precision (32/64), iterations (int))
experiments = [("CE", "Adam", 32, int(1e3)), # ("CE", "Adam", 64, int(1e2)),
               ("MSE", "Adam", 32, int(1e3)), ("MSE", "AMSGrad", 32, int(1e3))]

# If you run out of (cuda) memory, run one at a time individually or change device logic below.

# Above is purely for testing purposes
# This is the configuration used in the paper.
# experiments = [("CE", "Adam", 32, int(1e6)), ("CE", "Adam", 64, int(1e6)), ("MSE", "Adam", 32, int(1e6)), ("MSE", "AMSGrad", 32, int(1e6))]

# Be warned that raining the 64 bit resnet can take a very long time!
def start_train_loop(loss_fn="CE", optim="Adam", prec=32, its_end=int(1e2), device='cuda:0'):
    
    try:
        train_dataset = torchvision.datasets.CIFAR10(root='datasets', 
                                                train=True, 
                                                transform=transforms.ToTensor(),  
                                                download=False)
        
        # Data loader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, 
                                                    shuffle=True) 
            
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
            
        # Seed for the model
        seed = 42

        torch.manual_seed(seed)
        
        model = make_resnet18k(k=16, num_classes=10, bn=False)

        model = model.to(device)
        
        train_network(model, "CIFAR10", train_loader, loss_fn, optim, prec, its_end, device)
                    
    except Exception as e:
        with open("error_log.txt", "a") as f:
            f.write(f"Thread failed with error: {e}\n")

if __name__ == '__main__':
    mp.set_start_method('spawn')

    # will download the dataset if it does not exist yet so that does not happen in each subprocess.

    _ = torchvision.datasets.CIFAR10(root='datasets', 
                                                   train=True, 
                                                   transform=transforms.ToTensor(),  
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