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
    from pretreatment import rebin_xrf
except:
    sys.path.append('/jupyter/notebooks/XRF_AutoEncoder/utils/')
    from memory_utils import free_memory
    from pretreatment import rebin_xrf
    
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class XRFAE1DDataset(Dataset):
    """
        DATASET CLASS FOR 1D XRF AUTOENCODER
        
        Notice that here, X=Y.
    """
    def __init__(
        self, 
        # Opening Dataset
        xrf_path: str,              # path to H5DF dataset
        do_check: bool = True,      # Perform check to file, i.e. check if opened file contains "dataset_name"
        hist_name: str = 'hist',    # Name of the Histogram sector in HDF5 Dataset tree
        # Dataset Transform
        transform: type(lambda x:x) = None, # Method to trasform data
        max_size: int = -1,         # Max size of the dataset - MEMORY HANDLER UTILS
    ):
        # Dir to XRF files
        self.xrf_path = xrf_path  
        self.hist_name = hist_name
        
        self.max_size = max_size
        
        if not os.path.isfile(xrf_path):
            raise Exception(f"{xrf_path} is not a valid filepath.")
        # Transform method to be applied to each element
        self.transform = transform
        
        self._xrf = None
        # Open file
        with h5py.File(self.xrf_path, 'r') as _h5:
            # get keys, i.e. the filename list
            self._xrf_file_list = list( _h5.keys() )
            # iterate over filenames
            for idx, filename in enumerate(tqdm.tqdm(self._xrf_file_list) ):
                # get hist and hist only
                _temp = torch.Tensor( _h5[filename][self.hist_name][()] ).float()
                
                # NB: we may have created a dataset with single hists and not batched hists
                # check if this is the case, and then squeeze a dir
                if len( _temp.shape  ) == 1:
                    _temp = _temp.unsqueeze(0)
                
                # perform transform to avoid excess data leaks 
                # NB: it SLOWS DOWN the process
                # Apply transform
                if self.transform:
                    _temp = self.transform( _temp.to(device) ).cpu()#.detach()
                if idx == 0:
                    # Self xrf is none, so init
                    self._xrf = _temp
                else:
                    # Concatenate torch tensors
                    self._xrf = torch.cat( (self._xrf , _temp), dim=0)
                # Check if we have to break the loop for handling the memory size
                if self.max_size > 0 and self._xrf.shape[0] >= self.max_size:
                    print(f"Max Dataset size of {self.max_size} reached.\nReturning dataset of shape:\t{self._xrf.shape}")
                    break
                
                free_memory(to_delete=[_temp])
        
        
    def __len__(self):
        return self._xrf.shape[0]
    
    def __getitem__(self, idx: int):
        # nb: we use AutoEncoder, i.e. NO Y
        return self._xrf[idx, :]
    
    
class MAXRFVizDataset(Dataset):
    """
        DATASET CLASS FOR 2D MA-XRF AUTOENCODER trained model viz
        
        Notice that here, X=Y.
    """
    def __init__(
        self, 
        path_to_datacube: str, 
        data_name: str = 'img', 
        transform: type(lambda x:x) = None, # Method to trasform data
        MAX_BIN: int = 3000, 
        REBIN_SIZE: int = 1024, 
    ):
        """
        Init dataset
        
        Args:
            path_to_datacube (str)  : Path to datacube HDF5 file. 
            data_name        (str)  : Name od datacube in HDF5 file. Defaults to 'img', 
            transform        (func) : Transform method. None, # Method to trasform data
            MAX_BIN          (int)  : Max bin in Energy channel BEFORE rebinning. Defaults to 3000, 
            REBIN_SIZE       (int)  : Rebinning size; defaults to 1024, 
        """
        self.path_to_datacube = path_to_datacube
        self.transform = transform
        self.data_name = data_name
        self.REBIN_SIZE = REBIN_SIZE
        self.MAX_BIN = MAX_BIN
        
        with h5py.File(self.path_to_datacube, 'r') as _h5:
            # Open
            self.ma_xrf = torch.Tensor( np.array(_h5[self.data_name][()], dtype=float) ).float().nan_to_num(nan=0.0)
            # Remove zerps
            self.ma_xrf[self.ma_xrf < 0] = 0.0
            # Remove last bions
            self.ma_xrf = self.ma_xrf[:, :, :self.MAX_BIN]
            self.shape = self.ma_xrf.shape
            # Rebin 
            self.ma_xrf = rebin_xrf( self.ma_xrf.reshape(-1, self.shape[-1]), n_bins=self.REBIN_SIZE )
            #self.ma_xrf = self.ma_xrf.reshape(*self.shape[:2], self.REBIN_SIZE)
            # Transform
            if self.transform:
                self.ma_xrf = self.transform(self.ma_xrf)
            self.ma_xrf = self.ma_xrf.nan_to_num(nan=0.0)
            self.final_shape = self.ma_xrf.shape
            
    def __len__(self):
        return self.ma_xrf.shape[0]
    
    def __getitem__(self, idx: int):
        # nb: we use AutoEncoder, i.e. NO Y
        return self.ma_xrf[idx, :]
    
    
class AstroSynthDataset(Dataset):
    """
        DATASET CLASS FOR AstroSynthetic dataset for AUTOENCODER trained model 
        
        Notice that we have also a label.
        
        The files are in torch .pt format
    """
    def __init__(
        self, 
        path_to_data : str, 
        path_to_label: str, 
        transform: type(lambda x:x) = None, # Method to trasform data
        MAX_BIN   : int = 3000, 
        REBIN_SIZE: int = 1024, 
        max_size: int = -1,         # Max size of the dataset - MEMORY HANDLER UTILS
    ):
        """
        Init dataset
        
        Args:
            path_to_data     (str)  : Path to data in .pt format.
            path_to_label    (str)  : Path to label in .pt format.
            transform        (func) : Transform method. None, # Method to trasform data
            MAX_BIN          (int)  : Max bin in Energy channel BEFORE rebinning. Defaults to 3000, 
            REBIN_SIZE       (int)  : Rebinning size; defaults to 1024, 
        """
        self.path_to_data  = path_to_data
        self.path_to_label = path_to_label
        
        self.transform = transform
        self.REBIN_SIZE = REBIN_SIZE
        self.MAX_BIN    = MAX_BIN
        
        self.max_size = max_size         # Max size of the dataset - MEMORY HANDLER UTILS
        
        self._data   = torch.load(self.path_to_data)
        self._labels = torch.load(self.path_to_label) 
        
        if self.max_size > 0:
            self._data   = self._data[:self.max_size, : ]
            self._labels = self._labels[:self.max_size, : ]
            
        if self._data.shape[-1] > self.MAX_BIN:
            self._data = self._data[: self.MAX_BIN]
            
        if self._data.shape[-1] > self.REBIN_SIZE:
            self._data = rebin_xrf( self._data, n_bins=self.REBIN_SIZE )
        
        # Transform
        if self.transform:
            self._data = self.transform(self._data)
        self._data = self._data.nan_to_num(nan=0.0)
            
    def __len__(self):
        return self._data.shape[0]
    
    def __getitem__(self, idx: int):
        return self._data[idx, :], self._labels[idx, :]
    