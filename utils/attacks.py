import torch
import torch.nn.functional as F

class AutoScaleCrossEntropyLoss(torch.nn.Module):
    """
    A variant of CrossEntropyLoss that scales the inputs
    such that the maximum difference between logits does not exceed a certain maximum.
    """
    def __init__(self, max_logit=20.0, **kwargs):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(**kwargs)
        self.max_logit = max_logit
    def forward(self, input, target):
        with torch.no_grad():
            dx = torch.amax(input, 1, keepdim=True) - torch.amin(input, 1, keepdim=True)
            dx = torch.amax(dx)
            scale = torch.min(torch.tensor(1.0), self.max_logit / dx)
        return self.loss(input * scale, target)

class ScaledPGD(object):
    """
    Same as PGD below but uses AutoScaleCrossEntropyLoss above
    """
    def __init__(self, model,
                 eps=8/255,
                 alpha=2/255,
                 steps=10,
                 random_start=True,
                 dmax=None, dmin=None,
                 device='cuda'
                ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.device = device
        self.dmax = dmax
        self.dmin = dmin
            
    def __call__(self, images, labels):
        ## TODO: add targeted attacks
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device) 

        loss = AutoScaleCrossEntropyLoss()
        adv_images = images.clone().detach()
        dmin = images.min() if self.dmin is None else self.dmin
        dmax = images.max() if self.dmax is None else self.dmax

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.rand_like(adv_images)*2*self.eps - self.eps
            adv_images = torch.clamp(adv_images, min=dmin, max=dmax).detach()

        for _ in range(self.steps):
            
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=dmin, max=dmax).detach()

        return adv_images

class PGD(object):
    """
    Untargeted PGD attack. Modified from torchattack.attack to allow non-[0,1] domain data
    https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/pgd.html#PGD
    taken from Humayun et al for reproduction purposes, at: https://github.com/AhmedImtiazPrio/grok-adversarial/blob/main/attacks.py
    """
    def __init__(self, model,
                 eps=8/255,
                 alpha=2/255,
                 steps=10,
                 random_start=True,
                 dmax=None, dmin=None,
                 device='cuda'
                ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.device = device
        self.dmax = dmax
        self.dmin = dmin
            
    def __call__(self, images, labels):
        ## TODO: add targeted attacks
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device) 

        loss = torch.nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        dmin = images.min() if self.dmin is None else self.dmin
        dmax = images.max() if self.dmax is None else self.dmax

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.rand_like(adv_images)*2*self.eps - self.eps
            adv_images = torch.clamp(adv_images, min=dmin, max=dmax).detach()

        for _ in range(self.steps):
            
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=dmin, max=dmax).detach()

        return adv_images