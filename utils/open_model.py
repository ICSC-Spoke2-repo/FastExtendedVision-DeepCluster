import json 
import torch
from typing import Type

def load_model_func(model_name: str, _model_kwargs: dict, model_class: Type[ torch.nn.Module ], load_from_json: bool = True ):
    if load_from_json:
        with open(f'./{model_name}.json', 'r') as f:
            _stored_hyp = json.load(f)

        model_kwargs = { 
            key: _stored_hyp[key] 
                if key != 'final_activation'
                else  _model_kwargs[key] 
            for key in _model_kwargs.keys() 
        }
    else:
        model_kwargs = _model_kwargs
    # load the best model
    RELOAD_MODEL_NAME = f"./{model_name}.pth"
    checkpoint = torch.load(RELOAD_MODEL_NAME)
    print(f'Best model {RELOAD_MODEL_NAME} at epoch: ', checkpoint['epoch'])
    
    loaded_model = model_class(**model_kwargs)

    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval() 
    return loaded_model