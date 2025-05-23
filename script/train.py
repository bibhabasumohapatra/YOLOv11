import torch
from torch.utils.data import DataLoader
from dataset import COCODataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import Compose, ToTensor, Normalize

import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import torch
import tqdm
import yaml
from torch.utils import data
from ..model import DetectionModel
from utils import YOLOLoss
from utils import non_max_suppression

model = DetectionModel(nc=80)  # Replace with your actual model

# Define transformations
transform = Compose([
    ToTensor(),
])

# Load COCO validation dataset (subset of 5 images)
dataset = COCODataset(root='path/to/coco/val', annotation_file='path/to/annotations/instances_val2017.json', transform=transform)
subset = torch.utils.data.Subset(dataset, range(5))
dataloader = DataLoader(subset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Example optimizer

def train(args, model, dataloader, optimizer):
    # Move model to single GPU
    model = model.cuda()
    amp_scale = torch.amp.GradScaler()
    criterion = YOLOLoss()

    with open('weights/step.csv', 'w') as log:

        for epoch in range(args.epochs):
            model.train()
            # Remove distributed sampler epoch setting
            if args.epochs - epoch == 10:
                dataloader.dataset.mosaic = False

            p_bar = enumerate(dataloader)
            print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
            p_bar = tqdm.tqdm(p_bar, total=len(dataloader))

            optimizer.zero_grad()

            for i, (samples, targets) in p_bar:

                samples = samples.cuda().float()

                # Forward pass under autocast
                with torch.amp.autocast('cuda'):
                    outputs = model(samples)
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box_loss+=loss_box.item()
                avg_cls_loss+=loss_cls.item()
                avg_dfl_loss+=loss_dfl.item()

                loss_box *= args.batch_size
                loss_cls *= args.batch_size
                loss_dfl *= args.batch_size

                # Remove DDP gradient scaling, use world_size=1 for single GPU
                loss_box *= args.world_size
                loss_cls *= args.world_size
                loss_dfl *= args.world_size

                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                amp_scale.step(optimizer)
                amp_scale.update()
                optimizer.zero_grad()

                torch.cuda.synchronize()

                memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'
                s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                   avg_box_loss/len(dataloader), avg_cls_loss/len(dataloader), avg_dfl_loss/len(dataloader))
                p_bar.set_description(s)
