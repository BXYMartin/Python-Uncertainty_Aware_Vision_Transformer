#!/bin/bash
#SBATCH --gres=gpu:1
source /homes/mb220/.bashrc 
cd /vol/bitbucket/mb220/thesis-codebase
/vol/bitbucket/mb220/python-env/bin/python train.py --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --output_dir lidc-output-no-transform  --base_lr 0.05 --img_size 224  --batch_size 24 --dataset LIDC --root_path "./data/LIDC/"
