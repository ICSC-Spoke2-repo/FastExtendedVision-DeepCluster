import os 
# https://discuss.pytorch.org/t/cuda-launch-blocking-in-jupyter-notebook/163029
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import sys
import gc
from typing import Union, Type, Callable
import tqdm

import datetime
import time
import json 

import h5py
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import torch
import torchvision

import sys
sys.path.append('/jupyter/notebooks/DeepClusteringXRF/')
from utils.memory_utils import free_memory
from utils.logs_utils import write_line_to_file, store_hyp_dict
from utils.clustering_utils import silhouette_score

import plotly.graph_objects as go

def plot_clustered(
    ma_xrf: torch.Tensor, transformed_ma_xrf: torch.Tensor, reduced_ma_xrf: torch.Tensor,
    best_cl, best_c, best_score, best_K, s_scores, 
    _final_shape, 
    MIN_CLUSTER = 4, MAX_CLUSTER = 12,
    # add
    tech_name: str = 'TSNE' ,
    decoder = None, 
):
    """
    Function to plot the Clustered MA-XRF datacube.

    Args:
        ma_xrf              (torch.Tensor)  : MA-XRF datacube
        transformed_ma_xrf  (torch.Tensor)  : Transformed MA-XRF datacube
        reduced_ma_xrf      (torch.Tensor)  : Latent space representation of MA-XRF datacube
        best_cl             (torch.Tensor)  : Best Cluster Label tensor.
        best_c              (torch.Tensor)  : Best Cluster tensor
        best_score          (float)         : Best silhouette score
        best_K              (int)           : Number of clusters of the best K-Means
        s_scores            (list)          : List of all losses. 
        _final_shape        (list)          : MA-XRF datacube final shape
        MIN_CLUSTER         (int, optional) : IKmeans MIN_CLUSTER; for training history plot. Defaults to 4.
        MAX_CLUSTER         (int, optional) : IKmeans MAX_CLUSTER; for training history plot. Defaults to 12.
        tech_name           (str, optional) : Dim Red technique name. Defaults to 'TSNE'.

    Returns:
        fig1, fig2: Clustered fig, training fig. 
    """


    X = transformed_ma_xrf.reshape(-1, transformed_ma_xrf.shape[-1]).detach().cpu().numpy() 
    _ma_xrf = ma_xrf.detach().cpu().numpy().reshape(-1, ma_xrf.shape[-1] )

    N_features = ma_xrf.shape[-1]
    # ===== ITERATIVE KMeans ===================================================

    _scores = []

    USE_SKLEARN = False
    USE_PCA = False  

    _clustered = best_cl.detach().cpu().numpy()
    _cluster_centers = best_c.detach().cpu().numpy()
    _N_clusters = best_K#.detach().cpu().numpy()
    _scores_embed = _scores = s_scores.detach().cpu().numpy()

    # ===== CLUSTER PLOTS ===================================================
    clust_n_cols = 3 if tech_name == 'PCA' or tech_name == 'MLM' else 2
    add_thing = 1 if tech_name == 'PCA' or tech_name == 'MLM' else 0
    #clust_n_cols = 3 if  ma_xrf.sum() != transformed_ma_xrf.sum() else 2
    #add_thing = 1 if ma_xrf.sum() != transformed_ma_xrf.sum() else 0
    fig, ax = plt.subplots(nrows=_N_clusters, ncols=clust_n_cols, dpi=120, figsize=(20, 4*_N_clusters) )
    fig.suptitle(f'{tech_name}: Clustered MA-XRF with _N_clusters = {_N_clusters} -- Score: {best_score:.6f}\n\n.', fontsize=24)
    fig.tight_layout(w_pad=4)

    _N_clusters = _clustered.max()

    for idx, cluster in enumerate(range(0, _N_clusters+1)):
        _mask = np.copy(_clustered)
        if cluster>0:
            _mask[ _mask != cluster ] = 0
            _mask = _mask/cluster
        else:
            _mask[_mask > cluster ]  = -1
            _mask = _mask + 1
        # Imshow cluster
        ax[idx, 0].imshow(_mask.reshape(*_final_shape[:2]) )
        ax[idx, 0].set_title(f'Cluster {cluster}')

        
        if tech_name == 'PCA' or tech_name == 'MLM':
            # Rec average
            ax[idx, 1].imshow(
                np.array(_mask[:, None], dtype=bool) * transformed_ma_xrf.detach().cpu().numpy().reshape(-1, N_features), 
                aspect='auto', cmap='jet', origin='lower'
            )
            ax[idx, 1].set_title(f'Reconstructed XRF {cluster}-th Cluster Average')
            
            

            imax = ax[idx,1].twinx()
            imax.plot( 
                np.sum(
                    _mask[:, None] * transformed_ma_xrf.detach().cpu().numpy()  ,
                    axis=0
                ) / _mask.sum(),
                c='red' , label = 'Rec avg'
            )
            
            # if decoder is passed plot, also the transormed on the decoder
            if decoder is not None:
                _latent_avg  = np.sum(
                    decoder(
                        torch.tensor(
                            torch.from_numpy(_mask[:, None]).cpu() * reduced_ma_xrf.cpu() ,
                        ).float()
                    ).detach().cpu().numpy()  ,
                    axis=0
                )
                _latent_avg = _latent_avg - _latent_avg.min()
                _latent_avg = _latent_avg / _latent_avg.max()

                imax.plot( 
                    _latent_avg ,
                    c='gold' , label='Latent avg' , alpha = 0.8, linestyle='dashed',
                )
                imax.legend()
        # True average
        ax[idx, 1+add_thing].imshow(
            np.array(_mask[:, None], dtype=bool) * _ma_xrf.reshape(-1, N_features), 
            aspect='auto', cmap='jet', origin='lower'
        )
        imax2 = ax[idx,1+add_thing].twinx()
        imax2.plot( 
            np.sum(
                _mask[:, None] * ma_xrf.reshape(-1, N_features).detach().cpu().numpy()  ,
                axis=0
            ) / ( 1e-7 + _mask.sum() ),
            c='red'
        )
        ax[idx, 1+add_thing].set_title(f'True XRF {cluster}-th Cluster Average')

    
    plt.show()

    

    # ===== Clustering VIZ ===================================================
    #_N_clusters = _clustered.max()+1 

    rgb_list = np.array(
        [ 
            [
                c_idx, 
                min(  2 * c_idx , 2*( _clustered.max() -  c_idx ) ), #0,
                _clustered.max() - c_idx
            ]
            for c_idx in range(0,  _N_clusters + 1) ]
    ) / _clustered.max()

    expanded_clustered = np.zeros( (_clustered.shape[0], rgb_list.shape[0]) )
    for idx, cluster in enumerate(range(0, _N_clusters + 1 )):
        _mask = np.copy(_clustered)
        if cluster>0:
            _mask[_mask!=cluster ] = 0
            _mask = _mask/cluster
        else:
            _mask[_mask>cluster ]  = -1
            _mask = _mask + 1

        expanded_clustered[:, idx] = _mask


    # ===== HISTORY ===================================================
    fig2, ax2 = plt.subplots(dpi=120, figsize=(12,6))
    ax2.set_title(f"{tech_name}: Score in Iterative KMeans")
    ax2.scatter(np.arange(len(_scores)) + MIN_CLUSTER , _scores, label='Score in Whole space', c='red', marker = '+')
    ax2.set_xlabel('N clusters')
    ax2.set_ylabel('score in Embedding space', c='red')
    plt.show()
    
    return fig, fig2

def plot_histograms(
    ma_xrf: torch.Tensor, transformed_ma_xrf: torch.Tensor, reduced_ma_xrf: torch.Tensor,
    best_cl, best_c, best_score, best_K, s_scores, 
    _final_shape, 
    MIN_CLUSTER = 4, MAX_CLUSTER = 12,
    # add
    tech_name: str = 'TSNE' ,
    decoder = None, 
    plot_cmap: bool = False
):
    """
    Function to plot the Clustered MA-XRF Histrograms.

    Args:
        ma_xrf              (torch.Tensor)  : MA-XRF datacube
        transformed_ma_xrf  (torch.Tensor)  : Transformed MA-XRF datacube
        reduced_ma_xrf      (torch.Tensor)  : Latent space representation of MA-XRF datacube
        best_cl             (torch.Tensor)  : Best Cluster Label tensor.
        best_c              (torch.Tensor)  : Best Cluster tensor
        best_score          (float)         : Best silhouette score
        best_K              (int)           : Number of clusters of the best K-Means
        s_scores            (list)          : List of all losses. 
        _final_shape        (list)          : MA-XRF datacube final shape
        MIN_CLUSTER         (int, optional) : IKmeans MIN_CLUSTER; for training history plot. Defaults to 4.
        MAX_CLUSTER         (int, optional) : IKmeans MAX_CLUSTER; for training history plot. Defaults to 12.
        tech_name           (str, optional) : Dim Red technique name. Defaults to 'TSNE'.

    Returns:
        fig1: Histogram fig
    """
    
    X = transformed_ma_xrf.reshape(-1, transformed_ma_xrf.shape[-1]).detach().cpu().numpy() 
    _ma_xrf = ma_xrf.detach().cpu().numpy().reshape(-1, ma_xrf.shape[-1] )

    N_features = ma_xrf.shape[-1]
    # ===== ITERATIVE KMeans ===================================================

    _scores = []

    USE_SKLEARN = False
    USE_PCA = False  

    _clustered = best_cl.detach().cpu().numpy()
    _cluster_centers = best_c.detach().cpu().numpy()
    _N_clusters = best_K#.detach().cpu().numpy()
    _scores_embed = _scores = s_scores.detach().cpu().numpy()

    # ===== CLUSTER PLOTS ===================================================
    clust_n_cols = 2
    add_thing = 0
    #clust_n_cols = 3 if  ma_xrf.sum() != transformed_ma_xrf.sum() else 2
    #add_thing = 1 if ma_xrf.sum() != transformed_ma_xrf.sum() else 0
    fig, ax = plt.subplots(nrows=_N_clusters, ncols=clust_n_cols, dpi=120, figsize=(16, 4*_N_clusters) )
    fig.suptitle(f'{tech_name}: Clustered MA-XRF with _N_clusters = {_N_clusters} -- Score: {best_score:.6f}\n\n.', fontsize=24)
    fig.tight_layout(w_pad=2)

    _N_clusters = _clustered.max()

    for idx, cluster in enumerate(range(0, _N_clusters+1)):
        _mask = np.copy(_clustered)
        if cluster>0:
            _mask[ _mask != cluster ] = 0
            _mask = _mask/cluster
        else:
            _mask[_mask > cluster ]  = -1
            _mask = _mask + 1
        # Imshow cluster
        ax[idx, 0].imshow(_mask.reshape(*_final_shape[:2]) )
        ax[idx, 0].set_title(f'Cluster {cluster}')

        
        if tech_name == 'PCA' or tech_name == 'MLM':
            ax[idx, 1].set_title(f'XRF {cluster}-th Cluster Average')
            
            if plot_cmap:
                _cmap_true = np.array(_mask[:, None], dtype=bool) * _ma_xrf.reshape(-1, N_features)
                _cmapr_rec = np.array(_mask[:, None], dtype=bool) * transformed_ma_xrf.detach().cpu().numpy().reshape(-1, N_features)
                _cmap_diff = np.abs(
                    _cmapr_rec/_cmapr_rec.max() - _cmap_true/_cmap_true.max()
                )
                ax[idx,1].imshow(
                    _cmap_diff, 
                    aspect='auto', cmap='jet', origin='lower'
                )
                
                imax  = ax[idx,1].twinx()
                imax2 = imax #ax[idx,1+add_thing].twinx()
            else:
                imax  = ax[idx,1]
                imax2 = ax[idx,1+add_thing]
            imax.plot( 
                np.sum(
                    _mask[:, None] * transformed_ma_xrf.detach().cpu().numpy()  ,
                    axis=0
                ) / _mask.sum(),
                c='red' , label = 'Rec avg', linewidth=2
            )
            
            # if decoder is passed plot, also the transormed on the decoder
            if decoder is not None:
                _latent_avg  = np.sum(
                    decoder(
                        torch.tensor(
                            torch.from_numpy(_mask[:, None]).cpu() * reduced_ma_xrf.cpu() ,
                        ).float()
                    ).detach().cpu().numpy()  ,
                    axis=0
                )
                _latent_avg = _latent_avg - _latent_avg.min()
                _latent_avg = _latent_avg / _latent_avg.max()

                imax.plot( 
                    _latent_avg ,
                    c='gold' , label='Latent avg' , alpha = 0.6, linestyle='dotted', linewidth=1.5
                )
                
        # True average
        imax2.plot( 
            np.sum(
                _mask[:, None] * ma_xrf.reshape(-1, N_features).detach().cpu().numpy()  ,
                axis=0
            ) / ( 1e-7 + _mask.sum() ),
            c='green', label='True avg' , alpha = 0.8, linestyle='dashdot', linewidth=2
        )
    
        imax.legend()
    plt.show()
    
    return fig


# === 2D =========================================================================================

def plot_latent_space(
    ma_xrf: torch.Tensor, transformed_ma_xrf: torch.Tensor, reduced_ma_xrf: torch.Tensor,
    best_cl, best_c, best_score, best_K, s_scores, 
    _final_shape, 
    MIN_CLUSTER = 4, MAX_CLUSTER = 12,
    # add
    tech_name: str = 'TSNE',
    #
    N_CONTOUR_LEVELS: int = 10,
    PLOT_LATENT_SPACE_MA_XRF: bool = True,
    PLOT_HIST_LOG_Y_SCALE: bool = False, 
):
    """
    Function to plot the Clustered MA-XRF datacube.

    Args:
        ma_xrf              (torch.Tensor)  : MA-XRF datacube
        transformed_ma_xrf  (torch.Tensor)  : Transformed MA-XRF datacube
        reduced_ma_xrf      (torch.Tensor)  : Latent space representation of MA-XRF datacube
        best_cl             (torch.Tensor)  : Best Cluster Label tensor.
        best_c              (torch.Tensor)  : Best Cluster tensor
        best_score          (float)         : Best silhouette score
        best_K              (int)           : Number of clusters of the best K-Means
        s_scores            (list)          : List of all losses. 
        _final_shape        (list)          : MA-XRF datacube final shape
        MIN_CLUSTER         (int, optional) : IKmeans MIN_CLUSTER; for training history plot. Defaults to 4.
        MAX_CLUSTER         (int, optional) : IKmeans MAX_CLUSTER; for training history plot. Defaults to 12.
        tech_name           (str, optional) : Dim Red technique name. Defaults to 'TSNE'.
        N_CONTOUR_LEVELS    (int, optional): Number of contours in contour plot. Defaults to 10.
        PLOT_LATENT_SPACE_MA_XRF (bool, optional): If true, plots also Ma-XRF grayscale latent space direction image. Defaults to True.

    Returns:
        _type_: _description_
    """
    low_dim_mu = mu = reduced_ma_xrf.detach().numpy()
    
    _clustered = best_cl.detach().cpu().numpy()
    _cluster_centers = best_c.detach().cpu().numpy()
    _N_clusters = best_K#.detach().cpu().numpy()
    _scores_embed = _scores = s_scores.detach().cpu().numpy()

    #_score_embed = silhouette_score(reduced_ma_xrf.detach(), best_cl)
    #_score = silhouette_score(torch.Tensor(ma_xrf), best_cl)
    
    _min_pca_idx = 0
    
    # ===== Clustering VIZ ===================================================

    rgb_list = np.array(
        [ 
            [
                c_idx, 
                min(  2 * c_idx , 2*( _clustered.max() -  c_idx ) ), #0,
                _clustered.max() - c_idx
            ]
            for c_idx in range(0,  _N_clusters + 1) ],
        dtype = float
    ) / _clustered.max()
    #print(rgb_list)
    rgb_list = rgb_list if rgb_list.min() > 0. else rgb_list - rgb_list.min()    
    rgb_list = rgb_list/rgb_list.max()
    #print(rgb_list)
    
    expanded_clustered = np.zeros( (_clustered.shape[0], rgb_list.shape[0]) )
    for idx, cluster in enumerate(range(0, _N_clusters + 1 )):
        _mask = np.copy(_clustered)
        if cluster>0:
            _mask[_mask!=cluster ] = 0
            _mask = _mask/cluster
        else:
            _mask[_mask>cluster ]  = -1
            _mask = _mask + 1

        expanded_clustered[:, idx] = _mask
    
    N_COLS = mu.shape[-1] 
    N_ROWS = N_COLS
    add_col = 1 if PLOT_LATENT_SPACE_MA_XRF else 0
    fig, axs = plt.subplots(nrows=N_ROWS, ncols=N_COLS+add_col, dpi=150, figsize=(20,20) )

    fig.suptitle(f"{tech_name}: Clustered latent MA-XRF 2-D Projections w/ N_clusters = {_N_clusters+1} -- Score: {best_score:.6f}\n\n", fontsize=24)
    fig.tight_layout(w_pad=7, h_pad=3)

    for idx in range(N_ROWS):
        for jj in range(N_COLS+add_col):
            jdx = jj - 1 if PLOT_LATENT_SPACE_MA_XRF else jj
            if PLOT_LATENT_SPACE_MA_XRF and jj == 0:
                # IMSHOW OF LATENT SPACE DIRECTIONS
                ax = axs[idx, jj]
                try:
                    ax.imshow(reduced_ma_xrf.reshape(*_final_shape[:-1], -1)[:,:,idx], cmap='gray')
                    ax.set_axis_off() # remove axis
                    ax.set_title(f"MA-XRF on {tech_name}{idx}")
                except Exception as e:
                    print(f"i,j={(idx, jdx)}: {e}\n")
            else:
                # Scatter plots outside the PCA_x = PCA_y dir
                if idx != jdx:
                    ax = axs[idx, jj]

                    # scatter plot
                    if idx < jdx:
                        ax.scatter(
                            low_dim_mu[:, idx], 
                            low_dim_mu[:, jdx],
                            alpha=0.2,
                            c=expanded_clustered@rgb_list
                        )
                    # contour plot
                    else: 
                        for cluster_no in range(_N_clusters+1):
                            try:
                                _data_x = low_dim_mu[ np.array(expanded_clustered[:, cluster_no], dtype=bool) , idx]
                                _data_y = low_dim_mu[ np.array(expanded_clustered[:, cluster_no], dtype=bool) , jdx]
                                _range = np.array([
                                    [ _data_x.min(), _data_x.max() ],
                                    [ _data_y.min(), _data_y.max()]
                                ])
                                _N_contour_levels = N_CONTOUR_LEVELS if N_CONTOUR_LEVELS else 5
                                _N_bins = 30
                                delta_x = (_range[0,1] - _range[0,0])/_N_bins
                                delta_y = (_range[1,1] - _range[1,0])/_N_bins

                                x = np.arange( _range[0][0], _range[0][1] , delta_x)
                                y = np.arange( _range[1][0], _range[1][1] , delta_y)
                                X, Y = np.meshgrid(x, y)
                                Z, _, _ = np.histogram2d(
                                    _data_x, _data_y, 
                                    #range = _range,
                                    bins=[ len(x), len(y) ] 
                                )
                                _colors = [
                                    [*list(rgb_list[cluster_no]*step/_N_contour_levels) , step/_N_contour_levels ] for step in range(1, _N_contour_levels+1)
                                ]
                                CS = ax.contour(X, Y, Z.T, _N_contour_levels, colors = _colors)
                            except Exception as e:
                                #print(f"Contour exception: {e}")
                                pass

                    # add markers
                    _low_dim_cluster_centers = _cluster_centers
                    ax.scatter(
                        _low_dim_cluster_centers[:, idx], 
                        _low_dim_cluster_centers[:, jdx] ,
                        c = 'gold', 
                        marker = '+'
                    )
                    # add text to markers
                    for cluster_no, cluster_center in enumerate(_low_dim_cluster_centers):
                        ax.text( x = cluster_center[idx], y = cluster_center[jdx], s=cluster_no, color = 'black' , fontsize=13 )

                    ax.set_xlabel(f'{tech_name}{idx}', fontsize=14)
                    ax.set_ylabel(f'{tech_name}{jdx}', fontsize=14)

                # case PCA_x = PCA_y
                else:
                    ax = axs[idx, jj]

                    _hists = [
                        low_dim_mu[ np.array(expanded_clustered[:, cluster_no], dtype=bool) , idx] # right way
                        for cluster_no in range(0, _N_clusters + 1 )
                    ]
                    # There is improperly a number of zeros for each
                    #_hists = [ hist[hist != 0] for hist in _hists ]
                    ax.hist( 
                        _hists,
                        stacked=True,
                        bins = 100,
                        color = rgb_list,
                        label = [ f"{cluster_no}" for cluster_no in range(0, _N_clusters + 1 ) ]
                    )
                    ax.set_xlabel(f'{tech_name}{idx}'    , fontsize=14)
                    ax.set_ylabel(f'Total counts', fontsize=14)
                    ax.legend(fontsize= 10, loc='center left', bbox_to_anchor = (1, 0.5))
                    if PLOT_HIST_LOG_Y_SCALE:
                        ax.set_yscale('log', nonpositive='clip')

    # add text to fig
    _scale = 1/len(rgb_list)/2.
    for idx, color in enumerate(rgb_list):
        _string = f"{idx} - {np.round(color, 3)}"
        fig.text(x=1., y = 1 - _scale * (idx+3), s=_string, bbox=dict(fill=False, edgecolor=color, linewidth=2))

    plt.show()
    
    return fig

# === 3D =========================================================================================

def plot_clustered_3D(
    ma_xrf: torch.Tensor, transformed_ma_xrf: torch.Tensor, reduced_ma_xrf: torch.Tensor,
    best_cl, best_c, best_score, best_K, s_scores, 
    _final_shape, 
    MIN_CLUSTER = 4, MAX_CLUSTER = 12,
    # add
    tech_name: str = 'TSNE',
    dir_x : int = 0, dir_y : int = 1, dir_z : int = 2, 
    _alpha_scatter: float = 0.6,
    width: float = 500,
    height: float = 500,
):
    """
    Function to plot a 3D scatter plot along dir_x, dir_y and dir_z.

    Args:
        ma_xrf              (torch.Tensor)  : MA-XRF datacube
        transformed_ma_xrf  (torch.Tensor)  : Transformed MA-XRF datacube
        reduced_ma_xrf      (torch.Tensor)  : Latent space representation of MA-XRF datacube
        best_cl             (torch.Tensor)  : Best Cluster Label tensor.
        best_c              (torch.Tensor)  : Best Cluster tensor
        best_score          (float)         : Best silhouette score
        best_K              (int)           : Number of clusters of the best K-Means
        s_scores            (list)          : List of all losses. 
        _final_shape        (list)          : MA-XRF datacube final shape
        MIN_CLUSTER         (int, optional) : IKmeans MIN_CLUSTER; for training history plot. Defaults to 4.
        MAX_CLUSTER         (int, optional) : IKmeans MAX_CLUSTER; for training history plot. Defaults to 12.
        N_CONTOUR_LEVELS    (int, optional): Number of contours in contour plot. Defaults to 10.
        tech_name           (str, optional): _description_. Defaults to 'TSNE'.
        dir_x               (int, optional): _description_. Defaults to 0.
        dir_y               (int, optional): _description_. Defaults to 1.
        dir_z               (int, optional): _description_. Defaults to 2.
        _alpha_scatter      (float, optional): alpha value of scatter markers. Defaults to 0.6.
        width               (float, optional): Figure width. Defaults to 500,
        height              (float, optional): Figure height. Defaults to 500,
    Returns:
        _type_: _description_
    """
    N_features = ma_xrf.shape[-1]
    low_dim_mu = mu = reduced_ma_xrf.detach().numpy()
    
    _clustered = best_cl.detach().cpu().numpy()
    _cluster_centers = best_c.detach().cpu().numpy()
    _N_clusters = best_K#.detach().cpu().numpy()
    _scores_embed = _scores = s_scores.detach().cpu().numpy()

    #_score_embed = silhouette_score(reduced_ma_xrf.detach(), best_cl)
    #_score = silhouette_score(torch.Tensor(ma_xrf), best_cl)
    
    _min_pca_idx = 0
    
    # ===== Clustering VIZ ===================================================

    rgb_list = np.array(
        [ 
            [
                c_idx, 
                min(  2 * c_idx , 2*( _clustered.max() -  c_idx ) ), #0,
                _clustered.max() - c_idx
            ]
            for c_idx in range(0,  _N_clusters + 1) ],
        dtype = float
    ) / _clustered.max()
    #print(rgb_list)
    rgb_list = rgb_list if rgb_list.min() > 0. else rgb_list - rgb_list.min()    
    rgb_list = rgb_list/rgb_list.max()
    #print(rgb_list.shape)
    
    expanded_clustered = np.zeros( (_clustered.shape[0], rgb_list.shape[0]) )
    for idx, cluster in enumerate(range(0, _N_clusters + 1 )):
        _mask = np.copy(_clustered)
        if cluster>0:
            _mask[_mask!=cluster ] = 0
            _mask = _mask/cluster
        else:
            _mask[_mask>cluster ]  = -1
            _mask = _mask + 1

        expanded_clustered[:, idx] = _mask
    
    N_COLS = mu.shape[-1] 
    N_ROWS = N_COLS
    fig = go.Figure()
    #traces = []
    for clust_idx, _mask in enumerate(expanded_clustered.T):
        _col = rgb_list[clust_idx]
        _col = np.array(_col*255, dtype=int)
        _col = np.append(_col, _alpha_scatter) if _alpha_scatter > 0 and _alpha_scatter < 1 else _col
        #print(_col)
        np.array(_mask[:, None], dtype=bool) * transformed_ma_xrf.detach().cpu().numpy().reshape(-1, N_features)
        _only_clustered = np.array(_mask[:, None] , dtype=bool) * reduced_ma_xrf.detach().cpu().numpy()
        _trace = go.Scatter3d(
            x = _only_clustered[:,dir_x], y = _only_clustered[:,dir_y], z = _only_clustered[:,dir_z], 
            mode='markers' ,
            marker=dict(
                size=4,
                color=f'rgb{tuple(_col)}',                # set color to an array/list of desired values
                #colorscale='Viridis',   # choose a colorscale
                #opacity=0.8
            ),
            name = f"{clust_idx}th Cluster"
        )
        
        fig.add_trace(_trace)
        #traces.append(_trace)
    # update layout
    fig.update_layout(
        autosize=False,
        width=width,
        height=height
    )
    fig.show()
    return fig

def plot_3D_surfaces(
    data: np.array, 
    NBINS: str = 50 ,
    opacity: float = 0.2 ,
    surface_count: int = 10 ,
    width: float = 500,
    height: float = 500
):
    """
    Histgrammed surface plot. 

    Args:
        data                (np.array): (N_samples, 3) MA-XRF datacube
        NBINS               (str, optional): _description_. Defaults to 50.
        opacity             (float, optional): _description_. Defaults to 0.2.
        surface_count       (int, optional): _description_. Defaults to 10.
        width               (float, optional): Figure width. Defaults to 500,
        height              (float, optional): Figure height. Defaults to 500,

    Returns:
        _type_: _description_
    """
    
    H, edges = np.histogramdd(data, bins = NBINS)
    X, Y, Z = np.mgrid[
        :NBINS ,:NBINS, :NBINS
    ]

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=H.flatten(),
            isomin=0.2,
            #isomax=0.7,
            opacity=opacity ,
            surface_count=surface_count,
        )
    )

    # update layout
    fig.update_layout(
        autosize=False,
        width=width,
        height=height
    )

    fig.show()
    
    return fig