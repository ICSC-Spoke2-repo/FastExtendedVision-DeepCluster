from typing import Type, Union

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import scipy.stats as sts 

def hellinger_distance(p: torch.tensor, q: torch.tensor):
    return torch.sqrt(
        torch.sum(
            (
                torch.sqrt( p/p.sum() ) - torch.sqrt( q/q.sum() )
            ) ** 2
        )
    ) / torch.tensor(np.sqrt(2)).item() 


def linearize_alpha(_vect: torch.Tensor, _add_min: float = 0.1):
    """
    Function to define a mapping from weights to alphas (and linewidths such that extremal values are enhanched, i.e.
    y = ax^2 + bx + c 

    y(x_min) = y(x_max) = 1
    y(0) = 1
    
    to do so we write
    y(x) = a (x - x_min) (x-x_max) + 1
    
    where 
    a = -  1/(x_max * x_min) 
    """
    if len( _vect.shape ) > 1:
        raise Exception(f"Error: passed object is not a 1D vector")
    #return ( _vect - _vect.min() ) / ( _vect.max() - _vect.min() ) 
    _parabola =  - ( _vect - _vect.min() ) *  ( _vect - _vect.max() ) / ( _vect.max() * _vect.min()  ) +  1
    return np.clip( _parabola + _add_min  , a_min=0, a_max=1)




def plot_model(
    model_to_plot: Type[torch.nn.Module] ,   
    untrained_model: Union[Type[torch.nn.Module], None] = None, 
    # Args
    _N_Layer_stop: int = + float('inf') ,
    _N_hist_bins: int = 35,
    inset_height: int = 200,
    _dpi: int = 150, 
    _figsize: list = (24, 18), 
    _suptitle: str = f"Model weights",
    _suptitle_font: int = 24,
    _inset_title_font: int = 14,
    _inset_axis_font : int = 12,
    _colorbar_font_size : int = 16, 
    BASE_PATH_TO_STORE_PNG: str = '',
    MAGNIFYING_PERCENTAGE: int = 110, 
):

    MAX_INPUT_OUTPUT_SIZE = - float('inf')

    v_max = - float('inf')
    v_min = + float('inf')
    for name, param in model_to_plot.named_parameters():
        if 'weight' in f"{name}": 
            #print(name, param.size()) 
            v_max = max([v_max, param.max().item()])
            v_min = min([v_min, param.min().item()])
            MAX_INPUT_OUTPUT_SIZE = max([ torch.Tensor( list( param.size() ) ).max().item() , MAX_INPUT_OUTPUT_SIZE ])


    jet    = plt.get_cmap('jet') 
    cNorm  = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
    scalarMap = ScalarMappable(norm=cNorm, cmap=jet)


    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=_dpi, figsize = _figsize)
    fig.tight_layout()
    ax.set_title(_suptitle, fontsize = _suptitle_font)


    MIN_DIMS = int( MAX_INPUT_OUTPUT_SIZE ) * MAGNIFYING_PERCENTAGE//100 # MAGNIFYING_PERCENTAGE% of the max size
    MIN_DIMS = MIN_DIMS//2 # we plot below and above zero

    ax.set_ylim(- MIN_DIMS - inset_height, + MIN_DIMS + inset_height)
    ax.set_xticks([])
    ax.set_yticks([])

    _step = 0

    # Iterate over layers
    for name, param in model_to_plot.named_parameters():
        if 'weight' in f"{name}": 

            #print(f"name: {name},\tsize: {param.size()}")
            _weights = param.clone().detach().cpu()
            # transpose
            _weights = _weights.transpose(0,1)
            _n_node_l, _n_node_r = _weights.shape
            #if 'mu' in f"{name}" or 'var' in f"{name}": 
            # Set colors
            colors = scalarMap.to_rgba(_weights.numpy()).reshape(-1, 4)
            # SET ALPHA FOR WRIGHTS 
            _linearised_alpha = linearize_alpha( _weights.reshape(-1) ).numpy()
            colors[:, -1] = _linearised_alpha
            if 'var' in f"{name}": 
                pass
            else:
                # Plot lines
                try:
                    if _step <= _N_Layer_stop:
                        # ========================================
                        # plot lines
                    
                        # Compute segments
                        segments = []
                        for node_left in np.arange( - _n_node_l//2, +_n_node_l//2 ):
                            for node_right in np.arange( - _n_node_r//2, +_n_node_r//2 ):
                                segments.append([
                                    # X_0, Y_0
                                    [ _step , node_left ],
                                    # X_1, Y_1
                                    [ _step + 1, node_right]
                                ])
                        line_segments = LineCollection( segments, colors=colors , linewidths = 0.1 * _linearised_alpha ) #, alpha = 0.5)
                        ax.add_collection(line_segments)
                except Exception as e:
                    print(f"_step: {_step} {(name, param.size() )}\ncolors: {colors.shape}\nError: {e}\n")
                # ========================================
                # Plot nodes
                ax.scatter(
                    [ _step for _ in range(_n_node_l) ], # all located at _step
                    np.arange( - _n_node_l//2, +_n_node_l//2),
                    marker = 'o',
                    s = 4
                )

                # Add layer name
                delta_y_test = 10
                _text = f"{name}".split('.')[-2]
                ax.text(x=_step, y= delta_y_test + _n_node_l//2, s=_text, fontsize=14, horizontalalignment='center')


                if _step <= _N_Layer_stop:
                    # =======================================
                    # add weights means below
                    #_mean_text = f"weights mean\n{_weights.mean():5f}"
                    #ax.text(x=_step + 0.5, y= delta_y_test + (_n_node_l//2 + _n_node_r//2)/2, s=_mean_text, fontsize=14, horizontalalignment='center')

                    # add distribution inset axes
                    _delta_inset_x = 0.1
                    _delta_inset_y = 10 #* MIN_DIMS//300

                    axin = ax.inset_axes(
                        #bounds : [x0, y0, width, height] : Lower-left corner of inset axes, and its width and height.
                        [ _step + _delta_inset_x, - max(_n_node_l, _n_node_r)//2 - inset_height , 1 - 2* _delta_inset_x, inset_height - _delta_inset_y ] ,
                         # In case of specifying the tranfornm to be the data
                        transform=ax.transData         
                    )
                    axin.set_title(f"layer {_text} weights distribution", fontsize=_inset_title_font)
                    # Hide inset axis ticks
                    #axin.set_xticks([])
                    #axin.set_yticks([])
                    # Set 
                    axin.set_ylabel('counts', fontsize = _inset_axis_font)
                    axin.set_xlabel('value' , fontsize = _inset_axis_font)

                    #frequency, bins = np.histogram(, bins=50, range=[0, 100])
                    _hist, _bins = torch.histogram(_weights.reshape(-1), bins = _N_hist_bins)

                    _binst_step_type = 'mid'
                    axin.fill_between( _bins[:-1] , _hist, step =_binst_step_type , color = '#1f77b4', alpha = 0.6 )
                    axin.step(         _bins[:-1] , _hist, where=_binst_step_type , color = '#1f77b4', label='trained')

                    # plot distribution of counts
                    _x_range = np.arange(_bins.min(), _bins.max(), (_bins.max() - _bins.min() )/200  ) 
                    _mu  = _weights.reshape(-1).numpy().mean() 
                    _std = _weights.reshape(-1).numpy().std()
                    axin.plot(
                         _x_range,
                        _hist.max() * np.exp( - 1/2 * ( (_x_range - _mu)/_std )**2  ) ,
                        color = '#ff7f0e', # standard cycle colors https://matplotlib.org/stable/users/prev_whats_new/dflt_style_changes.html
                    )
                    # plot gauss fit
                    _std_rounded = np.round(_std, 4)
                    _mu_rounded  = np.round(_mu, 4)
                    # Non-gaussianity
                    _skewness = np.round( sts.skew( _weights.reshape(-1).numpy() )     , 2)
                    _kurtosis = np.round( sts.kurtosis( _weights.reshape(-1).numpy() ) , 2)
                    # Variation from initialisation
                    # in SNN weights are nornmalised such that mu=0 and mu_2 = 1
                    # in our implementation, 
                    # std(x) = 1 / sqrt(layer.out_features)
                    Delta_std = np.abs( _std - 1/_n_node_r )  
                    # if we want relative
                    #elta_std /= (1/_n_node_r )
                    np.round(  Delta_std , 3)
                    #_mu2 = np.round( torch.sum( _weights.reshape(-1)**2 ).numpy() / len( _weights.reshape(-1) ), 2)
                    
                    axin.text(
                        x=0.67, y=0.85, 
                        s= f"$\mu$ : {_mu_rounded:.4f}\n$\sigma$: {_std_rounded:.4f}\nskew: {_skewness}\nkurt: {_kurtosis}\n$\Delta \sigma_i$ : {Delta_std:.3f}", 
                        transform=axin.transAxes, 
                        fontdict={
                            'family': 'serif',
                            'color':  'darkred',
                            'weight': 'normal',
                            'size': 10,
                        },
                        verticalalignment = 'center' ,
                        # box
                        bbox=dict(fill=False, edgecolor='black', linewidth=0.6)
                    )
                    
                    # == Plots initialised model
                    try:
                        _untrained_weights = untrained_model.get_parameter(name).clone().detach().cpu()
                        _untrained_hist, _untrained_bins = torch.histogram(_untrained_weights.reshape(-1), bins = _bins) # USE THE SAME BINS OF TRAINED WEIGHTS
                        if _untrained_hist.sum() != _hist.sum():
                            print(f"Layer {name}\nUntrained sum: {_untrained_hist.sum()}\tTrained sum: {_hist.sum()}")
                        #_binst_step_type = 'mid'
                        axin.fill_between( _untrained_bins[:-1] , _untrained_hist, step =_binst_step_type , color = '#d62728', alpha = 0.3 )
                        axin.step(         _untrained_bins[:-1] , _untrained_hist, where=_binst_step_type , color = '#d62728', linestyle=':', label='untrained')
                        axin.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.075), fancybox=True, shadow=True, ncol=2)
                        try:
                            # Hellinger distance
                            hell_dist = hellinger_distance(_untrained_hist, _hist).item()
                            axin.text(
                                x=0.05, y=0.95, 
                                s= f"Hell dist: {hell_dist:.4f}", 
                                transform=axin.transAxes, 
                                fontdict={
                                    'family': 'serif',
                                    'color':  'darkblue',
                                    'weight': 'normal',
                                    'size': 10,
                                },
                                verticalalignment = 'center' ,
                                # box
                                bbox=dict(fill=False, edgecolor='black', linewidth=0.6)
                            )
                        except:
                            pass
                        pass
                    except: 
                        pass


                    # =======================================
                    # add weights imshow above
                    # add distribution inset axes
                    _delta_inset_x = 0.1
                    _delta_inset_y = 20 #* MIN_DIMS//300

                    axin2 = ax.inset_axes(
                        #bounds : [x0, y0, width, height] : Lower-left corner of inset axes, and its width and height.
                        [ _step + _delta_inset_x, + max(_n_node_l, _n_node_r)//2 + _delta_inset_y , 1 - 2* _delta_inset_x, inset_height ] ,
                         # In case of specifying the tranfornm to be the data
                        transform=ax.transData         
                    )

                    axin2.imshow(
                        _weights,
                        vmin=v_min, vmax = v_max, cmap='jet'
                    )
                    axin2.set_title(f"{_text} 2D weights view", fontsize=_inset_title_font)
                    axin2.set_xlabel("Out", fontsize = _inset_axis_font)
                    axin2.set_ylabel("In" , fontsize = _inset_axis_font)
                #print(_step)
                _step += 1
    # Add CMAP
    axcb = fig.colorbar(scalarMap)
    axcb.set_label('Weights', fontsize=_colorbar_font_size)
    
    return fig, ax
