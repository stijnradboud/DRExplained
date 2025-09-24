import torch as torch
import torchvision
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler, AdamW
import numpy as np
import gc
import matplotlib.pyplot as plt

from utils.LC.utils import flatten_model
from utils.LC.local_complexity import get_intersections_for_hulls
from utils.LC.samplers import get_ortho_hull_around_samples, get_ortho_hull_around_samples_w_orig
from torch.utils.data import Subset
import pickle

from tqdm import tqdm

'''
HERE ARE LOCAL COMPLEXITY PARAMETERS AND HELPER FUNCTIONS
'''

#@title Train and evaluation functions
## local complexity approx. parameters
config_approx_n = 1000                         # number of samples to use for approximation
config_n_frame = 25                            # number of vertices for neighborhood original: 40
config_r_frame = 0.005                         # radius of \ell-1 ball neighborhood  original: 0.005
config_LC_batch_size = 200
config_inc_centroid = False                     # include original sample as neighborhood vertex
config_seed = 42



def eval_LC(model, loaders, add_hook_fn, hulls=None):

    model, layer_names, activation_buffer = add_hook_fn(model)
    
    if hulls is not None:
        lcs = []
        for k in hulls.keys():
        
          # compute number of neurons that intersect hulls
          # using network activations
        
          with torch.no_grad():
        
            n_inters, _ = get_intersections_for_hulls(
                            hulls[k],
                            model=model,
                            batch_size=config_LC_batch_size,
                            layer_names=layer_names,
                            activation_buffer=activation_buffer
                      )
            lcs.append(n_inters)
    return lcs

def add_hooks(model, verbose=False):
    
    names,modules = flatten_model(model)
    assert len(names) == len(modules)
    
    ## add hooks to bns only. bn outputs are always passed through relus (even for skip connections)
    hook_module = torch.nn.modules.Linear
    layer_ids = np.asarray([i for i,each in enumerate(modules) if (type(each)==hook_module)])
    
    activation = {}
    def get_activation(name):
      def hook(model, input, output):
          activation[name] = output.detach()
      return hook
    
    for each in layer_ids:
      modules[each].register_forward_hook(get_activation(names[each]))

    if len(layer_ids > 0):
        layer_names = np.sort(np.asarray(names)[layer_ids])
    else:
        layer_names = []
    
    if verbose:
        print('Adding Hook to',layer_names)
    
    return model, layer_names, activation
    
def get_LC_samples(dloader, device):
    """
    Selects a set of samples for LC computation. TODO: allow subsampling classwise
    """

    samples = []
    labels = []

    size = 0
    for x,y in dloader:
        x = x.reshape(-1, 28*28).to(device)
        samples.append(x)
        labels.append(y)
        size += x.shape[0]
        if size >= config_approx_n: break

    ## concat and keep LC_batch_size
    samples = torch.concatenate(samples,axis=0)[:config_approx_n]
    labels = torch.concatenate(labels,axis=0)[:config_approx_n]

    return samples, labels

def perform_LC_analysis(checkpoints, ID, device):
    # MNIST dataset, 1K samples

    K = config_approx_n # enter your length here



    train_dataset = torchvision.datasets.MNIST(root='datasets', 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=False)

    test_dataset = torchvision.datasets.MNIST(root='datasets', 
                                            train=False, 
                                            transform=transforms.ToTensor(),
                                            download=False)

    # Load the indices
    with open('datasets/subsample_train_indices.pkl', 'rb') as f:
        subsample_train_indices = pickle.load(f)

    train_eval_subset = Subset(train_dataset, subsample_train_indices)
    # Data loader
    train_loader = torch.utils.data.DataLoader(train_eval_subset, batch_size=config_LC_batch_size, shuffle=False) 

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=config_LC_batch_size, 
                                            shuffle=False)
    ## initialize neighborhood sampler
    sampler_params = {'n' : config_n_frame if not config_inc_centroid \
                        else config_n_frame+1,
                    'r' : config_r_frame, 'seed':config_seed}

    sampler = get_ortho_hull_around_samples_w_orig if config_inc_centroid \
                else get_ortho_hull_around_samples

    ## select samples for neighborhood computation
    train_LC_batch, _ = get_LC_samples(train_loader, device)
    test_LC_batch, _ = get_LC_samples(test_loader, device)
    rand_LC_batch = torch.rand_like(test_LC_batch)   ## Data domain [0,1]


    ## sample hulls/neighborhoods
    train_hulls = sampler(
    train_LC_batch.cuda(),
    **sampler_params
        ).cpu()
    test_hulls = sampler(
        test_LC_batch.cuda(),
        **sampler_params
    ).cpu()
    rand_hulls = sampler(
        rand_LC_batch.cuda(),
        **sampler_params
    ).cpu()


    ## make hull dict. the keys of this dict will be used for logging
    ## you can add hulls separately for different classes as well to
    ## keep track of classwise statistics
    hulls = {
        'train' : train_hulls,
        'test' : test_hulls,
        'rand' : rand_hulls
    }

    loaders = {
        'train' : train_loader,
        'test' : test_loader
    }

    '''
    CALCULATE LCS IN EXPERIMENTS, CREATE PLOT FOR EACH
    '''

    train_lcs = []
    test_lcs = []
    random_lcs = []

   
    for i in tqdm(checkpoints):
        model = torch.load(f"checkpoints/models_MNIST/models_{ID}/model_at_{i}_its.ckpt", weights_only = False, map_location=torch.device('cuda'))
        model.to(torch.float32)
        model.eval()
        stats = eval_LC(model, loaders,
            hulls=hulls,
            add_hook_fn=add_hooks
            )
        train_lcs.append(stats[0].sum(1).mean(0).cpu().item())
        test_lcs.append(stats[1].sum(1).mean(0).cpu().item())
        random_lcs.append(stats[2].sum(1).mean(0).cpu().item())
        torch.cuda.empty_cache()
        gc.collect()

    plt.plot(checkpoints, train_lcs, label='train')
    plt.plot(checkpoints, test_lcs, label='test')
    plt.plot(checkpoints, random_lcs, label='random')
    plt.xscale("log")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("LC")
    plt.savefig(f"plots/Local Complexity - MNIST MLP - {ID}.png")
    plt.close()