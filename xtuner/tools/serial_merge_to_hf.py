import argparse
import os
import subprocess

import torch
from tqdm import tqdm


def merge_weights(ckpt_dir, new_ckpt_path):

    merged_weights = {}

    pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]

    for filename in tqdm(pt_files, desc='Merging weights'):

        file_path = os.path.join(ckpt_dir, filename)

        weights = torch.load(file_path, map_location='cpu')

        merged_weights.update(weights)

    torch.save(merged_weights, new_ckpt_path)
    return new_ckpt_path


def convert_to_hf(config_path, ckpt_path, output_dir):

    command = [
        'xtuner',
        'convert',
        'pth_to_hf',
        config_path,
        ckpt_path,
        output_dir,
    ]

    subprocess.run(command, check=True)


def process_weights(ckpt_dir, config_path, output_dir):
    # Step 1: Merge weights
    new_ckpt_filename = 'merged_model_states.pth'
    new_ckpt_path = os.path.join(ckpt_dir, new_ckpt_filename)
    merge_weights(ckpt_dir, new_ckpt_path)

    # Step 2: Convert to Hugging Face format
    convert_to_hf(config_path, new_ckpt_path, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Serial Merge Weights and Convert to Hugging Face Format')
    parser.add_argument(
        'ckpt_dir',
        type=str,
        help='The directory where the weight file is located')
    parser.add_argument(
        'config_path',
        type=str,
        help='Configuration file path used for training, \
        for example :work_dirs/**/epoch_3.pth ,\
        The directory is all in the file bf16_zero_pp_rank_*.pt. \
        which requires a xtuner convert merge first if it is qlora training.')
    parser.add_argument(
        'output_dir', type=str, help='Hugging Face model output directory')

    args = parser.parse_args()
    process_weights(args.ckpt_dir, args.config_path, args.output_dir)


if __name__ == '__main__':
    main()
