import json 

def write_line_to_file(LOG_FILE: str, log_line: str):
    with open(LOG_FILE, 'a') as f:
        f.write(log_line)
        
def store_hyp_dict(json_file: str, hyperparam_kwargs: dict, _indent: int = 4):
    with open(json_file, 'w') as fp:
        json.dump(hyperparam_kwargs, fp, indent=_indent)