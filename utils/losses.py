import os 

import sys
import gc
from typing import Union
import tqdm

import datetime
import time
import json 

import h5py

import numpy as np
import pandas as pd

import torch
import torchvision


# in torch/pytorch data and models need to be moved in the specific processing unit
# this code snippet allows to set the variable "device" according to available resoirce (cpu or cuda gpu)
if torch.cuda.is_available():
    print('number of devices: ', torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

import torch.nn as nn
import torch.nn.functional as F

try:
    from memory_utils import free_memory
    from clustering_utils import silhouette_score
except:
    sys.path.append('/jupyter/notebooks/XRF_AutoEncoder/utils/')
    from memory_utils import free_memory
    from clustering_utils import silhouette_score

# ======= MMD LOSS =======================================================================================

# Gaussian Kernel
def compute_kernel(x, y):
    """  
    The function compute_kernel(x, y) takes as input two matrices (x_size, dim) and (y_size, dim), 
    and returns a matrix (x_size, y_size) where the element (i,j) is the outcome of applying 
    the kernel to the i-th vector of x, and j-th vector of y.
    
    Notice that this is a MonteCarlo computation of the MMD. 
    """
    # Batch soze
    x_size = x.size(0)
    # Extracted number of samples; 256 is big enough for accurate prediction
    y_size = y.size(0)
    # latent space dim
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)
# MMD
def compute_mmd(x, y):
    x_kernel  = compute_kernel(x, x)
    y_kernel  = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def reconstruction_loss(x, x_hat):
    return (x_hat - x).pow(2).mean()

# ======= infoVAE LOSS =======================================================================================

def vae_loss_function(x_hat, x, z, β=1, monte_carlo_size: int = 256, return_all: bool = False):
    # to match samples from the prior p(z) and from the encoding distribution, 
    # we can simply generate samples from the prior distribution p(z), 
    # and compare the MMD distance between the real samples and the generated latent codes.
    #  256 is big enough for accurate MC approx-
    samples = torch.randn([ monte_carlo_size, z.shape[-1] ])
    # check device
    if z.is_cuda:
        samples = samples.to(device)
    mmd = compute_mmd(samples, z)
    
    rec_loss = reconstruction_loss(x, x_hat)
    
    infoVAE_loss = rec_loss + β * mmd
    
    if return_all:
        return infoVAE_loss, rec_loss,  mmd
    return infoVAE_loss

# ======= SILHOUETTE LOSS =======================================================================================

def silhouette_loss(feats: torch.Tensor, labels: torch.Tensor):
    return ( 1 - silhouette_score(feats, labels) )/2

# ======= TOTAL LOSS =======================================================================================

def compute_total_loss(loss, avg_s_loss, γ, use_sum = False):
    if use_sum:
        return loss + γ * avg_s_loss
    return loss*(1  + γ * avg_s_loss)/(1+γ)