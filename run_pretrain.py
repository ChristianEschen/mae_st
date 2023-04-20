# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from pathlib import Path

from main_pretrain import get_args_parser, main, read_yaml
import yaml

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    # load config yaml file
    config = read_yaml(args.config_path)
    args.output_dir = config['output_dir']
    
    
    if config['output_dir']:
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    main(args)
