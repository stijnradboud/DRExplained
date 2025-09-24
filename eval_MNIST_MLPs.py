from utils.eval_utils import get_checkpoints, check_dead_neurons, get_metrics, get_SC_results, plot_metrics
from utils.LC_utils import perform_LC_analysis
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import pickle
from torch.utils.data import SubsetRandomSampler
'''
This file creates all the plots in the paper for all the MNIST MLP experiments.

Below is the same list of experiments as in the train file for MNIST MLPs.

Update accordingly based on the experiments that were ran.
'''
experiments = [("CE", "Adam", 32, int(1e3)), ("CE", "Adam", 64, int(1e3)),
               ("MSE", "Adam", 32, int(1e3)), ("MSE", "AMSGrad", 32, int(1e3))]

# Again, if you run out of (cuda) memory, run one at a time individually or change device logic below.

# This is the configuration used in the paper.
# experiments = [("CE", "Adam", 32, int(1e6)), ("CE", "Adam", 64, int(1e6)), ("MSE", "Adam", 32, int(1e7)), ("MSE", "AMSGrad", 32, int(1e7))]

# You can turn on individual analyses/plots below:
analyse_general_metrics = True
analyse_local_complexity = True
analyse_dead_neurons = True
analyse_softmax_collapse = True

def evaluate_and_plot_checkpoints(loss_fn = "CE", optim="Adam", prec=32, its_end=int(1e3), device=torch.device("cuda:0")):
 
    if prec == 32:
        image_dtype = torch.float32
    elif prec == 64:
        image_dtype = torch.float64
    else:
        print("Unsupported precision!")
        return -1

    train_dataset = torchvision.datasets.MNIST(root='datasets', 
                                           train=True, 
                                           transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Lambda(lambda x: x.view(-1))
                                                    ]),
                                           download=True)

    test_dataset = torchvision.datasets.MNIST(root='datasets', 
                                            train=False, 
                                            transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Lambda(lambda x: x.view(-1))
                                                    ]),
                                            download=True)

    with open('datasets/subsample_train_indices.pkl', 'rb') as f:
        subsample_train_indices = pickle.load(f)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, 
                                           sampler=SubsetRandomSampler(subsample_train_indices), shuffle=False) 
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    
    ID = f"{loss_fn}_{optim}_{prec}"

    model_dir = f"checkpoints/models_MNIST/models_{ID}"

    print(f"Evaluating {ID}...")
    checkpoints = get_checkpoints(its_end)

    if analyse_general_metrics:
        print("\tAccuracies, losses and adversarial accuracies...")
        metrics = get_metrics(model_dir, checkpoints, train_loader, test_loader, device, loss_fn=loss_fn, prec=prec)

        plot_metrics(metrics, checkpoints, loss_fn, f"MNIST MLP - {ID}")

    if analyse_local_complexity:
        print(f"\tLC Analysis...")
        perform_LC_analysis(checkpoints, ID, device)

    dead_neurons = []
    if analyse_dead_neurons or (analyse_softmax_collapse and loss_fn == "CE"):
        print(f"\tDead Neurons and/or SC...")
        underflows = []
        absorps = []
        all0 = []
        for its in tqdm(checkpoints):
            model = torch.load(f"{model_dir}/model_at_{its}_its.ckpt", weights_only=False, map_location=device)
            model.eval()
            if analyse_softmax_collapse and loss_fn == "CE":
                n_underflow, n_absorp, n_all = get_SC_results(model, train_loader, prec, device)
                underflows.append(n_underflow)
                absorps.append(n_absorp)
                all0.append(n_all)
            if analyse_dead_neurons:
                _, sum = check_dead_neurons(model, train_loader, prec)
                dead_neurons.append(sum / 2 / 4) # 4 layers of each 200 * 100

        if analyse_softmax_collapse and loss_fn == "CE":
            plt.plot(checkpoints, underflows, label='Only Underflow Soft-Max Collapse errors', color='blue')
            plt.plot(checkpoints, absorps, label='Only Absorpion Soft-Max Collapse errors', color='orange')
            plt.plot(checkpoints, all0, label='Gradient completly zero', color='red')
            plt.grid()
            plt.xscale("log")
            plt.xlabel("Optimizer iterations")
            plt.ylabel("%")
            plt.ylim(-5, 105)
            plt.legend()
            plt.savefig(f"plots/Softmax Collapse - MNIST MLP - {ID}.png")
            plt.close()

    return ID, checkpoints, dead_neurons

if __name__ == '__main__':
    results_dead_neurons = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for loss_fn, optim, prec, its_end in experiments:
        ID, checkpoints, dead_neurons = evaluate_and_plot_checkpoints(loss_fn, optim, prec, its_end)

        results_dead_neurons.append((ID, checkpoints, dead_neurons))

    if analyse_dead_neurons:
        plt.figure(figsize=(10,3))

        for ID, checkpoints, dead_neurons in results_dead_neurons:
            plt.plot(checkpoints, dead_neurons, label = ID)

        plt.legend()
        plt.xscale("log")
        plt.xlabel("Optimizer Iterations")
        plt.ylabel("% Dead Neurons across Network")
        plt.ylim(0, 100)

        plt.savefig("plots/Dead Neurons - MNIST MLPs.png", dpi=300, bbox_inches='tight')

        plt.close()