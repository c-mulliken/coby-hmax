import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # model args
    parser.add_argument('--out_dim', type=int, default=65536)
    parser.add_argument('--last_norm', type=bool, default=True)
    parser.add_argument('--momentum_teacher', type=float, default=0.996)

    # teacher params
    parser.add_argument('--warmup_teacher_temp', type=float, default=0.04)
    parser.add_argument('--teacher_temp', type=float, default=0.04)
    parser.add_argument('--warmup_teacher_epochs', type=int, default=0)

    # training params
    parser.add_argument('--use_fp16', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--weight_decay_end', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--freeze_last_layer', type=int, default=1)
    
    # data augmenation params
    parser.add_argument('--global_crops_scale', type=float, nargs=+, default=(0.4, 1.))
    parser.add_argument('--local_crops_number', type=int, default=8)
    parser.add_argument('--local_crops_scale', type=float, nargs=+, default=(0.05, 0.4))

    # misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='~/.cache/kagglehub/datasets/arjunashok33/miniimagenet/versions/1')
    parser.add_argument('--output_dir', type=str, default='output')

    return parser

def train_dino(args):
    student = models.resnet50()
    teacher = models.resnet50()

    transform = ...
    train_dataset = datasets.ImageFolder(root='~/.cache/kagglehub/datasets/arjunashok33/miniimagenet/versions/1',
                                         transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
