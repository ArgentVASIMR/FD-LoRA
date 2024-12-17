import utils
from training import train
from parser import get_parser
import argparse

#TODO:
# - Add a mode to convert a yaml stack to a single yaml modifier file
# - Add a mode to convert a yaml stack into a toml file for direct sd-scripts usage

def main():
    parser = get_parser()
    args = parser.parse_args()
    yaml_list = args.yaml_list
    arg_dict = utils.stack_yamls(yaml_list,'yamls')
    train(arg_dict, args.script_dir)

if __name__ == "__main__":
    main()