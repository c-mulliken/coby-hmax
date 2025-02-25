import argparse
import utils

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import DataLoader
from PIL import Image

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
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.))
    parser.add_argument('--local_crops_number', type=int, default=8)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4))

    # misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='~/.cache/kagglehub/datasets/arjunashok33/miniimagenet/versions/1')
    parser.add_argument('--output_dir', type=str, default='output')

    return parser

class DataAugment:
    def __init__(self, global_crops_scale, local_crops_number, local_crops_scale):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2)
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.global_trans1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize
        ])

        self.global_trans2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize
        ])

        


def train_dino(args):
    student = models.resnet50()
    teacher = models.resnet50()

    transform = ...
    train_dataset = datasets.ImageFolder(root='~/.cache/kagglehub/datasets/arjunashok33/miniimagenet/versions/1',
                                         transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
