import yaml
import os
def read_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)
def write_yaml(path: str, dict: dict) -> None:
    with open(path, 'w') as f:
        yaml.dump(dict, f)
def stack_yamls(paths: list[str],root_dir: str = None) -> dict:
    out = {}
    if root_dir is not None:
        paths = [os.path.join(root_dir,path) for path in paths]
    for path in paths:
        out.update(read_yaml(path))
    return out