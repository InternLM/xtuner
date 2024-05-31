import os
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

from xtuner.model.utils import guess_load_checkpoint
import shutil
import argparse


def convert_phi_to_official(phi_path, trained_path, save_path):
    statue_dict = guess_load_checkpoint(trained_path)
    # print(statue_dict.keys())
    print('================================================')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    shutil.copytree(phi_path, save_path)

    files = [f for f in os.listdir(phi_path) if f.endswith('safetensors')]

    for file in tqdm(files, desc='Convert'):
        tensors = {}
        new_path = os.path.join(save_path, file)
        old_path = os.path.join(phi_path, file)

        with safe_open(old_path, framework='pt', device='cpu') as f:
            for key in f.keys():
                find = False
                for trained_k, trained_v in statue_dict.items():
                    trained_k = trained_k[6:]
                    if key == trained_k:
                        tensors[key] = trained_v
                        find = True
                        break
                if not find:
                    tensors[key] = f.get_tensor(key)
            # print(f.keys())
            metadata = f.metadata()
            save_file(tensors, new_path, metadata=metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig-phi-path', '-o',
                        default='/mnt/hwfile/xtuner/huanghaian/model/Mini-InternVL-Chat-4B-V1-5')
    parser.add_argument('--trained-path', '-t',
                        default='/mnt/petrelfs/huanghaian/code/xtuner/work_dirs/mini_internvl_phi3_sft/iter_200.pth')
    parser.add_argument('--save-path', '-s',
                        default='/mnt/hwfile/xtuner/huanghaian/model/finetune_Mini-InternVL-Chat-4B-V1-5')
    args = parser.parse_args()
    convert_phi_to_official(args.orig_phi_path, args.trained_path, args.save_path)


if __name__ == '__main__':
    main()
