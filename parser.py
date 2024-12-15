import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-s','--script_dir', type=str, help='SD scripts directory')
    parser.add_argument('-y','--yaml_list', type=str, nargs='+', help='YAML files to parse')
    return parser