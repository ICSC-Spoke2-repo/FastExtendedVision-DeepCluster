import gc
import torch

def free_memory(to_delete: list):
    """Method to free CUDA memory
    
    Args:
        to_delete (list) : list of the variable names to be deleted
    """
    for _var in to_delete:
        try:
            del _val
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()