import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset, LIDC_dataset
from utils import test_single_volume, test_multiple_image
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./data/LIDC/', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='LIDC', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_LIDC', help='list dir')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1234, help='random seed')

args = parser.parse_args()

def inference(args, model, test_save_path=None):
    
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in (bar := tqdm(enumerate(testloader))):
        if len(sampled_batch["image"].shape) == 3:
            sampled_batch["image"] = sampled_batch["image"].unsqueeze(0)
        
        if len(sampled_batch["label"].shape) == 3:
            sampled_batch["label"] = sampled_batch["label"].unsqueeze(0)
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_multiple_image(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, repeat=12)
        metric_list += np.array(metric_i)
        bar.set_description('idx %d case %s mean_ged %f mean_ncc %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list * args.batch_size / len(db_test)
    print(metric_list)
    logging.info('Testing performance in best val model: mean_ged %f mean_ncc %f' % (metric_list[0][0], metric_list[0][1]))
    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'LIDC': {
            'Dataset': LIDC_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_LIDC',
            'num_classes': 2,
            'z_spacing': 1,
            'is_volume': False,
        }
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_volume = dataset_config[dataset_name]['is_volume']
    args.is_pretrain = True

    db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    count = 0
    for i_batch, sampled_batch in enumerate(testloader):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        selected = True
        for item in label:
            if np.isclose(item.max(), 0, 1e-8):
                selected = False
                break
        if selected:
            print(case_name)
            # count += 1
    # print(count)


