import mappings
import os
import sys
import yaml
import utils


def generate_flags(dict: dict) -> list[str]:
    out = []
    for k, v in dict.items():
        if isinstance(v, bool):
            if v:out.append(f"--{k}")
            continue
        if isinstance(v, str):
            if v == "": continue
        if isinstance(v, list):
            out.append(f"--{k}")
            [out.append(x) for x in v]
            continue
        out.append(f"--{k}={v}")
    return out

def train(arg_dict: dict, sd_scripts_install: str):
    old_work_dir = os.getcwd()
    os.chdir(sd_scripts_install)

    sys.path.append(sd_scripts_install)
    
    #if it's stupid and it works, then it's stupid
    import train_network as tn
    import sdxl_train_network as tn_sdxl
    import library.train_util as train_util
    
    sdxl = arg_dict['sdxl']

    if sdxl: 
        lib = tn_sdxl
        trainer = tn_sdxl.SdxlNetworkTrainer()
    else: 
        lib = tn
        trainer = tn.NetworkTrainer()
    mapper = mappings.mapper(arg_dict['config'])
    mapper.preprocess_config()
    parser = lib.setup_parser()
    flags = generate_flags(arg_dict)
    args = parser.parse_args(flags)
    train_util.verify_command_line_training_args(args)
    print('training')
    trainer.train(args)
    os.chdir(old_work_dir) 
