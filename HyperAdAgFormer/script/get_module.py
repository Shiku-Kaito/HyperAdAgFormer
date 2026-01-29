import argparse
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from utils import *
from dataloader  import *
from torchvision.models import resnet18, resnet34, resnet50

from HyperAdAgFormer_script.train import train_net as HyperAdAgFormer_script_train_net
from HyperAdAgFormer_script.network import HyperAdAgFormer
from HyperAdAgFormer_script.eval import eval_net as HyperAdAgFormer_script_eval_net

from losses import *

def get_module(args):
    if args.module ==  "HyperAdAgFormer":
        args.mode = "HyperAdAgFormer" 
        # Dataloader
        train_loader, val_loader, test_loader = load_data_bags(args) 
        # Model
        model = HyperAdAgFormer(args=args, num_classes=args.classes, embed_dim=512, depth=args.transfomer_layer_num, train_loader=train_loader)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        loss_function = nn.BCELoss()
        # Train & evaluation net
        train_net = HyperAdAgFormer_script_train_net
        eval_net = HyperAdAgFormer_script_eval_net

    else:
        print("Module ERROR!!!!!")

    return train_net, eval_net, model, optimizer, loss_function, train_loader, val_loader, test_loader