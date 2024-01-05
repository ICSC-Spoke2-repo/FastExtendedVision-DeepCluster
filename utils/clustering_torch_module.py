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

class DC_IterativeKMeans(nn.Module):
    def __init__(
        self, 
        min_n_cluster: int = 5, 
        max_n_cluster: int = 10, 
        Niter: int = 10, 
        verbose: bool = False, 
        random_centroid_init: bool = False
    ):
        super().__init__()
        
        self.min_n_cluster = min_n_cluster
        self.max_n_cluster = max_n_cluster if max_n_cluster > min_n_cluster else min_n_cluster
        self.Niter   = Niter if Niter > 1 else 1
        self.verbose = verbose
        self.random_centroid_init = random_centroid_init
        
    def forward(self, x):
        best_cl, best_c, best_score, best_K, s_scores = self.IterativeKMeans(
            x, 
            min_n_cluster = self.min_n_cluster, max_n_cluster = self.max_n_cluster, Niter = self.Niter, 
            verbose = self.verbose, random_centroid_init = self.random_centroid_init
        )
        return best_cl, best_c, best_score, best_K, s_scores
    
    
    
    #== METHODS ====================
    def silhouette_score(self, feats: torch.Tensor, labels: torch.Tensor) -> float:
        device, dtype = feats.device, feats.dtype
        unique_labels = torch.unique(labels)
        num_samples   = len(feats)
        if not (1 < len(unique_labels) < num_samples):
            raise ValueError(f"num unique labels must be > 1 and < num samples\nlen(unique_labels) = {len(unique_labels)}\tnum_samples = {num_samples}\n")
        scores = []

        for L in unique_labels:
            curr_cluster = feats[labels == L]
            num_elements = len(curr_cluster)

            if num_elements > 1:
                intra_cluster_dists = torch.cdist(curr_cluster, curr_cluster)
                mean_intra_dists = torch.sum(intra_cluster_dists, dim=1) / (
                    num_elements - 1
                )  # minus 1 to exclude self distance
                dists_to_other_clusters = []

                for otherL in unique_labels:
                    if otherL != L:
                        other_cluster = feats[labels == otherL]
                        inter_cluster_dists = torch.cdist(curr_cluster, other_cluster)
                        mean_inter_dists = torch.sum(inter_cluster_dists, dim=1) / (
                            len(other_cluster)
                        )
                        dists_to_other_clusters.append(mean_inter_dists)
                dists_to_other_clusters = torch.stack(dists_to_other_clusters, dim=1)
                min_dists, _ = torch.min(dists_to_other_clusters, dim=1)
                curr_scores  = (min_dists - mean_intra_dists) / (
                    torch.maximum(min_dists, mean_intra_dists)
                )
            else:
                curr_scores = torch.tensor([0], device=device, dtype=dtype)

            scores.append(curr_scores)

        scores = torch.cat(scores, dim=0)
        if len(scores) != num_samples:
            raise ValueError(
                f"scores (shape {scores.shape}) should have same length as feats (shape {feats.shape})"
            )
        return torch.mean(scores).item()
        
        
    def kpp_init(self, x: torch.Tensor, K: int) -> torch.Tensor:
        """ 
        Kmeans++ initialisation.
        See "k-means++:  The Advantages of Careful Seeding", D. Arthur and S. Vassilvitskii, http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
        See also: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_kmeans.py

        "We propose a specific way of choosing centers for thek-meansalgorithm. 
        In particular, let D(x) denote the shortest distance from a data point to the closest center we have already chosen. Then, we define the following algorithm, which we callk-means++:

            1a. Take one center c_1, chosen uniformly at random from X
            1b. Take a new center c_i, choosing x\in X with probability  D(x)^2/\sum D(x)2.
            1c. Repeat Step 1b. until we have taken k centers altogether
        """
        # 0. ini
        _device = x.device
        N, D = x.shape  # Number of samples, dimension of the ambient space
        _x = x.clone().to(_device)
        c  = torch.zeros_like(_x[:K, :]).to(_device)

        # 1a. get random vector
        idx_max = torch.randint(_x.shape[0], size = [1] )
        c[0] = _x[idx_max]
        # 1b. Compute the max distance vector and iterate
        for _step in range(1, K):
            # 1. Compute D
            x_i = _x.view(N, 1, D).expand(N, _step, D).to(_device)
            c_j =  c[:_step].view(1, _step, D).expand(N, _step, D).to(_device) # (1, _step, D) centroids
            _D = ((x_i - c_j) ** 2).sum(-1) # (N, _step)  distances
            _D = _D.min(dim=-1).values # min on centroids, i.e. D(x) = min_i dist(x, c_i) , (N, )
            # 2. Compute p
            probability_distribution = _D.pow(2) / _D.pow(2).sum()
            # 3. Random extraction
            # See https://discuss.pytorch.org/t/sampling-from-a-tensor-in-torch/97112
            # https://pytorch.org/docs/stable/generated/torch.multinomial.html#torch.multinomial
            idx_max = probability_distribution.multinomial(num_samples = 1, replacement=True)
            c[_step] = _x[idx_max]
        #print(f'K: {K} kpp')
        return c
    
    #
    def KMeans(self, x: torch.Tensor, K: int = 10, Niter: int = 10, verbose: bool = False, random_centroid_init: bool = False):
        """Implements Lloyd's algorithm for the Euclidean metric."""
        if K < 2:
            raise Exception(f"N_clusters has to be bigger that 2; inserted value: {K}")
        if verbose: 
            start = time.time()
        _device = x.device

        N, D = x.shape  # Number of samples, dimension of the ambient space
        if random_centroid_init:
            c = self.kpp_init(x, K).to(_device)
            # Add gaussian noise
            #c = c + x.std() * torch.randn( c.shape ).to(_device)
        else:
            c = x[:K, :].clone()  # Simplistic initialization for the centroids

        c.to(_device)
        original_c = c.clone() # internal debugging

        x_i = x.view(N, 1, D).expand(N, K, D).to(_device) # (N, 1, D) samples
        c_j = c.view(1, K, D).expand(N, K, D).to(_device) # (1, K, D) centroids

        cl = None
        old_cl = None
        # K-means loop:
        # - x  is the (N, D) point cloud,
        # - cl is the (N,) vector of class labels
        # - c  is the (K, D) cloud of cluster centroids
        for i in range(Niter):
            if type(cl) == torch.Tensor:
                old_cl = cl.clone()
            # E step: assign points to the closest cluster -------------------------
            D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
            cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

            # panic mode: check if cl has the right amounts of centroids, else break loop
            if len( cl.unique() ) < K and type(old_cl) == torch.Tensor:
                cl = old_cl.clone() # previous step
                break
            # M step: update the centroids to the normalized cluster average: ------
            # Compute the sum of points per cluster:
            c.zero_()
            c.scatter_add_(0, cl[:, None].repeat(1, D), x)

            # Divide by the number of points per cluster:
            Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1).to(_device)
            c  /= Ncl  # in-place division to compute the average

        # Check on clusters
        if len( cl.unique() ) < K:
            # =================================================================
            # Do something
            
            raise Exception(f"Error;\tK: {K};\tuniq: {len( cl.unique() )}")

        return cl, c
    
    def IterativeKMeans(self, x: torch.Tensor, min_n_cluster: int = 5, max_n_cluster: int = 10, Niter: int = 10, verbose: bool = False, random_centroid_init: bool = False):
        _device = x.device

        if max_n_cluster < min_n_cluster:
            max_n_cluster = min_n_cluster
        # init silhuette scores to MINIMUM, i.e. -1
        s_scores = - torch.ones(max_n_cluster + 1 - min_n_cluster) 
        s_scores = s_scores.to(_device)

        # Init return values
        best_K = -1
        best_cl= torch.zeros(x.shape[0])
        best_c = torch.zeros(x.shape[1]).unsqueeze(0)
        best_score = -1

        for idx, K in enumerate( range(min_n_cluster, max_n_cluster+1) ):
            try:
                res = self.KMeans(
                    x, 
                    K = K, Niter = Niter, random_centroid_init = random_centroid_init, verbose = verbose
                )
                cl, c = res
                #print(f'{cl.shape}, {c.shape}')
                score = self.silhouette_score(x, cl)
                if score > s_scores.max():
                    best_K  = K
                    best_cl = cl
                    best_c  = c
                    best_score = score

                s_scores[idx] = score
            except Exception as e:
                if verbose:
                    print(e)
                pass

        return best_cl, best_c, best_score, best_K, s_scores

