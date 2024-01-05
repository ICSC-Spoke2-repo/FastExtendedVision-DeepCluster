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
except:
    sys.path.append('/jupyter/notebooks/XRF_AutoEncoder/utils/')
    from memory_utils import free_memory
    
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import math
from collections import OrderedDict
from torchsummary import summary

# ====================== Canonical NEural Network =================================================================================
class DNN(nn.Module):
    def __init__(
        self, 
        in_dim: int,      # Input dimension
        out_dim: int,     # Output Dimension
        hidden_dims: list # Hidden Layer Dims
    ):
        super(DNN, self).__init__()
        self.dropout_prob = 0.1
        
        self.n_layers = len(hidden_dims)
        
        layers = OrderedDict()
        for i in range(self.n_layers):
            # Input layer
            if i == 0:
                layers[f"fc{i}"] = nn.Linear(in_dim, hidden_dims[0], bias=False)
            # Hidden Layer(s)
            else:
                layers[f"fc{i}"] = nn.Linear(hidden_dims[i-1], hidden_dims[i], bias=False)
            layers[f"relu_{i}"] = nn.ReLU()
            layers[f"dropout_{i}"] = nn.Dropout(p=self.dropout_prob)
        
        # Output Layers
        layers[f"fc_{i+1}"] = nn.Linear(hidden_dims[-1], out_dim, bias=True)
        
        self.network = nn.Sequential(layers)
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.network:
            if not isinstance(layer, nn.Linear):
                continue
            nn.init.normal_(layer.weight, std=1 / math.sqrt(layer.out_features) )
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)
                
    def forward(self, x):
        return self.network(x)

# ====================== Self Normalising NEural Network =================================================================================

class SNN(nn.Module):
    def __init__(
        self, 
        in_dim: int,      # Input dimension
        out_dim: int,     # Output Dimension
        hidden_dims: list # Hidden Layer Dims
    ):
        super(SNN, self).__init__()
        self._lambda = 1.0507
        self._alpha  = 1.6733
        self.dropout_prob = 0.1
        
        self.n_layers = len(hidden_dims)
        
        layers = OrderedDict()
        for i in range(self.n_layers):
            # Input layer
            if i == 0:
                layers[f"fc{i}"] = nn.Linear(in_dim, hidden_dims[0], bias=False)
            # Hidden Layer(s)
            else:
                layers[f"fc{i}"] = nn.Linear(hidden_dims[i-1], hidden_dims[i], bias=False)
            layers[f"selu_{i}"] = nn.SELU()
            layers[f"dropout_{i}"] = nn.AlphaDropout(p=self.dropout_prob)
        
        # Output Layers
        layers[f"fc_{i+1}"] = nn.Linear(hidden_dims[-1], out_dim, bias=True)
        
        self.network = nn.Sequential(layers)
        self.reset_parameters()

    def forward(self, x):
        return self.network(x)

    def reset_parameters(self):
        for layer in self.network:
            if not isinstance(layer, nn.Linear):
                continue
            nn.init.normal_(layer.weight, std=1 / math.sqrt(layer.out_features) )
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)

    def track_layer_activations(self, x):
        activations = []
        for layer in self.network:
            x = layer.forward(x)
            if isinstance(layer, nn.SELU):
                activations.append(x.data.flatten())
        return activations


# ====================== NEW =================================================================================
# Split Class    
class SplitActivation(nn.Module):
    def __init__(
        self, 
        latent_space_dim: int,
        use_latent_space_activation: bool = False # Latent space different activation YES/NO
    ):
        super(SplitActivation, self).__init__()
        self.latent_space_dim = latent_space_dim
        self.use_latent_space_activation = use_latent_space_activation
        if self.use_latent_space_activation:
            # We want to map μ ∈ [-1, +1] and σ ∈ [0,2] i.e. log σ ∈ [- ∞, 0]
            # This is since the KLD is comprised by two part
            # KLD = f(σ) + g(μ),
            # where
            # f(σ) = log σ - σ - 1 ,  g(μ) = μ^2
            # the minimum of g is at μ=0, while the one of f is at σ=1

            # So, for μ we apply the HardTanh https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html#torch.nn.Hardtanh
            self.mu_activation = nn.Hardtanh() # = F.hardtanh(mu)
            # while for the log we can either use the minus leaky RELU ( https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU )
            # or the minus ReLU6 ( https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html#torch.nn.ReLU6 )
            #logvar = - F.leaky_relu(-logvar)
            #logvar = - F.relu6(-logvar) / 3
            self.logvar_activation = nn.LeakyReLU()
        
        
    def forward(self, mu_logvar):
        mu_logvar = mu_logvar.view(-1, 2, self.latent_space_dim)
        mu     = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        # if use normal activation, pass activation function
        if self.use_latent_space_activation:
            # 
            mu = self.mu_activation(mu)
            # recall we need to apply minus two times
            logvar = - self.logvar_activation( - logvar )
        
        # Else, normal return
        return mu, logvar
    
# =========================================================
# VAE
# =========================================================
# Encoder
class VAEencoder(nn.Module):
    def __init__(
        self, 
        input_dim: int = 512,         # the number of energy channels
        n_layers : int = 6,           # Hidden layers for both Encoder AND Decoder
        encoding_space_dim: int = 64, # Encoding space dimension, i.e. tje space befor the mu, sigma layer
        latent_space_dim: int = 10,   # Latent space dimensions NB: the Z space; so the Encoder Output will be 2*latent_space_dim dimensional
        use_latent_space_activation: bool = False, # Latent space different activation YES/NO
        pow_2_decrease: bool = False, # If true, decrease by a Power of two the internal dimensions (512 -> 256 -> 128 -> etc.)
        use_SNN: bool = True,         # If true, Self-Normalising Neural networks are used
    ):
        super(VAEencoder, self).__init__()
        # Attrs
        self.input_dim = input_dim
        self.n_layers  = n_layers
        self.latent_space_dim = latent_space_dim
        self.encoding_space_dim = encoding_space_dim
        self.use_latent_space_activation = use_latent_space_activation
        
        self.use_SNN = use_SNN
        
        if pow_2_decrease:
            self.hidden_dims = [ 
                input_dim//(2**n)
                for n in range(1, n_layers)
                if self.input_dim//(2**n) > self.encoding_space_dim
            ]
        else:
            self.hidden_dims = [ 
                input_dim//n
                for n in range(2, n_layers+1)
                if self.input_dim//n > self.encoding_space_dim
            ]
        
        _enc_net = SNN(
            in_dim  = self.input_dim,           # Input dimension
            out_dim = self.encoding_space_dim,  # Output Dimension
            hidden_dims = self.hidden_dims
        ) if self.use_SNN else DNN(
            in_dim  = self.input_dim,           # Input dimension
            out_dim = self.encoding_space_dim,  # Output Dimension
            hidden_dims = self.hidden_dims
        )
        
        self.encoder = nn.Sequential(
            _enc_net,
            nn.Tanh()
        )
        
        # distribution parameters
        self.fc_mu  = nn.Linear(self.encoding_space_dim, self.latent_space_dim)
        self.fc_var = nn.Linear(self.encoding_space_dim, self.latent_space_dim)
        # Latetn space act
        self.use_latent_space_activation = use_latent_space_activation
        if self.use_latent_space_activation:
            # We want to map μ ∈ [-1, +1] and σ ∈ [0,2] i.e. log σ ∈ [- ∞, 0]
            # This is since the KLD is comprised by two part
            # KLD = f(σ) + g(μ),
            # where
            # f(σ) = log σ - σ - 1 ,  g(μ) = μ^2
            # the minimum of g is at μ=0, while the one of f is at σ=1

            # So, for μ we apply the HardTanh https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html#torch.nn.Hardtanh
            self.mu_activation = nn.Hardtanh() # = F.hardtanh(mu)
            # while for the log we can either use the minus leaky RELU ( https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU )
            # or the minus ReLU6 ( https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html#torch.nn.ReLU6 )
            #logvar = - F.leaky_relu(-logvar)
            #logvar = - F.relu6(-logvar) / 3
            self.logvar_activation = nn.LeakyReLU()
            
    def forward(self, x):
        x = self.encoder(x)
        # mu
        mu     = self.fc_mu(x)
        # logvar
        logvar = self.fc_var(x)
        
        # if use normal activation, pass activation function
        if self.use_latent_space_activation:
            # 
            mu = self.mu_activation(mu)
            # recall we need to apply minus two times
            logvar = - self.logvar_activation( - logvar )
        
        # Else, normal return
        return mu, logvar

# Decoder    
class VAEdecoder(nn.Module):
    def __init__(
        self, 
        input_dim: int = 512,         # the number of energy channels
        n_layers : int = 6,           # Hidden layers for both Encoder AND Decoder
        latent_space_dim: int = 10,   # Latent space dimensions NB: the Z space; so the Encoder Output will be 2*latent_space_dim dimensional
        final_activation: type( nn.Sigmoid() ) = nn.Sigmoid(), # Final activation
        pow_2_increase: bool = False,  # If true, increase by a Power of two the internal dimensions (128 -> 256 -> 512 -> etc.)
        use_SNN: bool = True,          # If true, Self-Normalising Neural networks are used
    ):
        super(VAEdecoder, self).__init__()
        # Attrs
        self.input_dim = input_dim
        self.n_layers  = n_layers
        self.latent_space_dim = latent_space_dim
        self.final_activation = final_activation
        
        self.use_SNN = use_SNN
        
        if pow_2_increase:
            self.hidden_dims = [
                input_dim//(2**n)
                for n in range(n_layers, 1, -1)
                if input_dim//(2**n) < self.input_dim
            ]
        else:
            self.hidden_dims = [
                input_dim//n
                for n in range(n_layers, 1, -1)
                if input_dim//n < self.input_dim
            ]
        
        _dec_nec = SNN(
            in_dim  = self.latent_space_dim,
            out_dim = self.input_dim,
            hidden_dims = self.hidden_dims
        ) if self.use_SNN else DNN(
            in_dim  = self.latent_space_dim,
            out_dim = self.input_dim,
            hidden_dims = self.hidden_dims
        )
        # NB: step 2 is 512//(2+1) = 170, NOT 512//2^n = 128
        self.decoder = nn.Sequential(
            OrderedDict([
                (
                    'Dec_SNN', 
                    _dec_nec
                ),
                (
                    'Dec_act', self.final_activation
                )
            ])            
        )
        
    def forward(self, z):
        return self.decoder(z) 
    
# Full VAE   
class VAE1D(nn.Module):
    def __init__(
        self, 
        input_dim: int = 512,         # the number of energy channels
        n_layers : int = 6,           # Hidden layers for both Encoder AND Decoder
        encoding_space_dim: int = 64, # Encoding space dimension (before FC layers for mu, logvar
        latent_space_dim: int = 10,   # Latent space dimensions NB: the Z space; so the Encoder Output will be 2*latent_space_dim dimensional
        use_latent_space_activation: bool = False, # Latent space different activation YES/NO
        final_activation: type( nn.Sigmoid() ) = nn.Sigmoid(), # Final decoder layer activation 
        use_log_scale: bool = False,  # Add additional float learnable paramete
        pow_2_decrease: bool = False, # If true, decrease by a Power of two the internal dimensions (512 -> 256 -> 128 -> etc.)
        pow_2_increase: bool = False, # If true, increase by a Power of two the internal dimensions (128 -> 256 -> 512 -> etc.)
        use_SNN: bool = True,         # If true, Self-Normalising Neural networks are used
    ):
        super(VAE1D, self).__init__()
        
        # Attrs
        self.input_dim = input_dim
        self.n_layers  = n_layers
        
        self.encoding_space_dim = encoding_space_dim
        self.latent_space_dim   = latent_space_dim
        
        self.use_latent_space_activation = use_latent_space_activation
        self.final_activation = final_activation
        
        self.use_log_scale = use_log_scale
        
        self.pow_2_decrease = pow_2_decrease
        self.pow_2_increase = pow_2_increase
        
        self.use_SNN = use_SNN
        
        # Coefficient for beta-vae
        self._is_vae = True
        
        # ENCODER
        self.encoder = VAEencoder(
            input_dim = self.input_dim,        # the number of energy channels
            n_layers  = self.n_layers,          # Hidden layers for both Encoder AND Decoder
            encoding_space_dim = self.encoding_space_dim,   # Encoding space dimension (before FC layers for mu, logvar
            latent_space_dim   = self.latent_space_dim,     # Latent space dimensions NB: the Z space; so the Encoder Output will be 2*latent_space_dim dimensional
            use_latent_space_activation = self.use_latent_space_activation, # Latent space different activation YES/NO
            pow_2_decrease = self.pow_2_decrease,    # If true, decrease by a Power of two the internal dimensions (512 -> 256 -> 128 -> etc.)
            use_SNN = self.use_SNN,                  # If true, Self-Normalising Neural networks are used
        )
        
        #  Decoder
        self.decoder = VAEdecoder(
            input_dim = self.input_dim,      # the number of energy channels
            n_layers  = self.n_layers,       # Hidden layers for both Encoder AND Decoder
            latent_space_dim = self.latent_space_dim,  # Latent space dimensions NB: the Z space; so the Encoder Output will be 2*latent_space_dim dimensional
            final_activation = self.final_activation,  # 
            pow_2_increase = self.pow_2_increase,      # If true, increase by a Power of two the internal dimensions (128 -> 256 -> 512 -> etc.)
            use_SNN = self.use_SNN,                    # If true, Self-Normalising Neural networks are used
        )
        
        # for the gaussian likelihood
        if self.use_log_scale:
            self.log_scale = nn.Parameter(torch.Tensor([0.0]))
    
    # reparametrisation trick
    def reparameterise(self, mu, logvar):
        if self.training and self._is_vae:
            return VAE1D.compute_z(mu, logvar)
        else:
            return mu
    
    # ========================================================
    # Forward
    def forward(self, x):
        # Encoder
        mu, logvar = self.encoder(x)
        # Reparametrisation trick
        z = self.reparameterise(mu, logvar)
        # Decoder
        return self.decoder(z), z, mu, logvar
    
    # ==========================================================
    # Static methods
    @staticmethod 
    def compute_z(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.new_empty(std.size()).normal_()
        return eps.mul_(std).add_(mu)
        # std = torch.exp(logvar / 2)
        # q = torch.distributions.Normal(mu, std)
        # z = q.rsample()
        # return z

# ==============================================================================
# Full Deep Clustering VAE
try:
    from clustering_torch_module import DC_IterativeKMeans
except:
    sys.path.append('/jupyter/notebooks/XRF_AutoEncoder/utils/')
    from clustering_torch_module import DC_IterativeKMeans
    
class DeepClustering_VAE1D(nn.Module):
    def __init__(
        self, 
        input_dim: int = 512,         # the number of energy channels
        n_layers : int = 6,           # Hidden layers for both Encoder AND Decoder
        encoding_space_dim: int = 64, # Encoding space dimension (before FC layers for mu, logvar
        latent_space_dim: int = 10,   # Latent space dimensions NB: the Z space; so the Encoder Output will be 2*latent_space_dim dimensional
        use_latent_space_activation: bool = False, # Latent space different activation YES/NO
        final_activation: type( nn.Sigmoid() ) = nn.Sigmoid(), # Final decoder layer activation 
        use_log_scale: bool = False,  # Add additional float learnable paramete
        pow_2_decrease: bool = False, # If true, decrease by a Power of two the internal dimensions (512 -> 256 -> 128 -> etc.)
        pow_2_increase: bool = False, # If true, increase by a Power of two the internal dimensions (128 -> 256 -> 512 -> etc.)
        # Deep Clustering Part
        min_n_cluster: int = 5,       # Minimal Number of clusters for IterativeKMeans
        max_n_cluster: int = 10,      # Maximal Number of clusters for IterativeKMeans
        Niter: int = 10,              # Number of KMeans iterations
        verbose: bool = False,        # Verbosity
        random_centroid_init: bool = False, # Use Random or Kpp Algo for initialisation. 
        use_SNN: bool = True,         # If true, Self-Normalising Neural networks are used
    ):
        super().__init__()
        
        # Attrs
        self.input_dim = input_dim
        self.n_layers  = n_layers
        
        self.encoding_space_dim = encoding_space_dim
        self.latent_space_dim   = latent_space_dim
        
        self.use_latent_space_activation = use_latent_space_activation
        self.final_activation = final_activation
        
        self.use_SNN = use_SNN
        
        self.use_log_scale = use_log_scale
        
        self.pow_2_decrease = pow_2_decrease
        self.pow_2_increase = pow_2_increase
        
        # For Deep Clustering
        self.min_n_cluster = min_n_cluster
        self.max_n_cluster = max_n_cluster if max_n_cluster > min_n_cluster else min_n_cluster
        self.Niter   = Niter if Niter > 1 else 1
        self.verbose = verbose
        self.random_centroid_init = random_centroid_init
        # Init IterativeKmenasClustering algo
        self.IKMeans = DC_IterativeKMeans(
            min_n_cluster = self.min_n_cluster, max_n_cluster = self.max_n_cluster, Niter = self.Niter, 
            verbose = self.verbose, random_centroid_init = self.random_centroid_init
        )
        
        # Coefficient for beta-vae
        self._is_vae = True
        # Coefficient for DeepCLustering
        self._is_deepclustering = True
        
        # ENCODER
        self.encoder = VAEencoder(
            input_dim = self.input_dim,        # the number of energy channels
            n_layers  = self.n_layers,          # Hidden layers for both Encoder AND Decoder
            encoding_space_dim = self.encoding_space_dim,   # Encoding space dimension (before FC layers for mu, logvar
            latent_space_dim   = self.latent_space_dim,     # Latent space dimensions NB: the Z space; so the Encoder Output will be 2*latent_space_dim dimensional
            use_latent_space_activation = self.use_latent_space_activation, # Latent space different activation YES/NO
            pow_2_decrease = self.pow_2_decrease,   # If true, decrease by a Power of two the internal dimensions (512 -> 256 -> 128 -> etc.)
            use_SNN = self.use_SNN,                 # If true, Self-Normalising Neural networks are used
        )
        
        #  Decoder
        self.decoder = VAEdecoder(
            input_dim = self.input_dim,      # the number of energy channels
            n_layers  = self.n_layers,       # Hidden layers for both Encoder AND Decoder
            latent_space_dim = self.latent_space_dim,  # Latent space dimensions NB: the Z space; so the Encoder Output will be 2*latent_space_dim dimensional
            final_activation = self.final_activation,  # 
            pow_2_increase = self.pow_2_increase,      # If true, increase by a Power of two the internal dimensions (128 -> 256 -> 512 -> etc.)
            use_SNN = self.use_SNN,                    # If true, Self-Normalising Neural networks are used
        )
        
        # for the gaussian likelihood
        if self.use_log_scale:
            self.log_scale = nn.Parameter(torch.Tensor([0.0]))
    
    # reparametrisation trick
    def reparameterise(self, mu, logvar):
        if self.training and self._is_vae:
            return self.compute_z(mu, logvar)
        else:
            return mu
    
    # ========================================================
    # Forward
    def forward(self, x):
        # Encoder
        mu, logvar = self.encoder(x)
        # Reparametrisation trick
        z = self.reparameterise(mu, logvar)
        # Deep Clustering
        if self._is_deepclustering:
            best_cl, best_c, best_score, best_K, s_scores = self.IKMeans(mu)
        else:
            best_cl, best_c, best_score, best_K, s_scores = self.IKMeans(mu)     ## <========== TB implemented ======================00
        # Decoder
        return self.decoder(z), z, mu, logvar, best_cl, best_c, best_score, best_K, s_scores
    
    # ==========================================================
    # Static methods
    def compute_z(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.new_empty(std.size()).normal_()
        return eps.mul_(std).add_(mu)
        # std = torch.exp(logvar / 2)
        # q = torch.distributions.Normal(mu, std)
        # z = q.rsample()
        # return z