import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

def rebin_xrf(
    img: torch.Tensor, 
    n_bins: int = 512
) -> torch.Tensor:
    _original_hist_size = img.shape[-1]
    divisor = int(np.ceil(_original_hist_size // n_bins) ) #_original_hist_size // n_bins
    #rebinning
    if divisor > 1:
        _device = img.device
        
        rebinned = torch.zeros(
            img.shape[0], n_bins,
            device=_device
        )
        
        # check device
        rebinned.to(_device)

        for step in range(divisor):
            _temp = img[
                :,
                # start : end : step
                step : _original_hist_size : divisor
            ]
            _temp.to(_device)
            rebinned += _temp[:, :n_bins  ]

        return rebinned
    # no rebinning
    else:
        return img
    
def custom_transform(
    x: torch.Tensor, 
    regulator: float = 10e-9, 
    q_low_value: float = 0.25, 
    q_high_value: float = 0.75,
    ignore_zeros: bool = False,
    final_transform: str = 'tanh'
) -> torch.Tensor:
    _x = torch.clone(x)
    if ignore_zeros:
        # Compute quantiles ignoring zeros; 
        # to do so, we replace 0 with nan, and use torch.nanquantile
        _x[_x == 0] = torch.nan
    
    # 1. compute log
    _mean = x.nanmean(dim=-1)
    if len(x.shape) == 2:
        _mean = _mean[:, None]
    elif len(x.shape) == 3: 
        _mean = _mean[:, :, None]
    
    y = torch.log(
        1 + (_x - _mean)/(_mean + regulator) + regulator 
    )
    # 2. Quantile transform
    _Q1 = y.nanquantile(q=q_low_value , dim=-1)
    _Q3 = y.nanquantile(q=q_high_value, dim=-1)
    _alpha = 2/(_Q3 - _Q1)
    _beta  = - (_Q3 + _Q1)/(_Q3 - _Q1)
    # Replace the NAN value with the appropriate values by simply computing the one withot the NaN replacement
    y = torch.log(
        1 + (x - _mean)/(_mean + regulator) + regulator 
    )
    
    #print(f"Q1={_Q1}, Q3={_Q3}")
    # expand a e b
    if len(y.shape) == 2:
        _alpha = _alpha[:,None]
        _beta  = _beta[:, None] 
    elif len(y.shape) == 3:
        _alpha = _alpha[:,:,None]
        _beta  = _beta[:,:, None] 
    z = _alpha * y + _beta
    # 3 final transfo
    out = torch.sigmoid(z) if final_transform == 'sigmoid' else torch.tanh(z) 
    if torch.isnan(out).any():
        try:
            out = torch.nan_to_num(out, nan = np.nanmin( out.numpy() ) )
        except: 
            _device = out.get_device()
            out = torch.nan_to_num(out, nan = np.nanmin( out.cpu().numpy() ) )
            out.to(_device)
    return out

def normalize_hist(_x: torch.Tensor) -> torch.Tensor:
    _x = _x / torch.max(_x, dim=-1)[0].unsqueeze(-1)
    return _x.nan_to_num(nan=0.0)

def tanh_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(
        x - x.mean(dim=-1)
    )

def smooth_1d(inputs: torch.tensor,  kernel_size: int = 5) -> torch.tensor:
    if len(inputs.shape) == 2:
        inputs = inputs.unsqueeze(1) 
    weights = torch.ones(kernel_size, dtype=inputs.dtype)/kernel_size
    weights = weights.unsqueeze(0).unsqueeze(0)
    return torch.nn.functional.conv1d(inputs, weights, padding = 'same').squeeze(1)