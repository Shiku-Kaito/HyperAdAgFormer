import os
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch.nn.functional as F
from statistics import mean
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from skimage import io
import glob
import torchvision.transforms as T
from torch.utils.data import WeightedRandomSampler
import random
import copy
from utils import *


#toy
class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, patient_names, tables, labels):
        np.random.seed(args.seed)
        self.patient_names = patient_names
        self.tables = tables
        self.labels = labels
        self.len = len(self.patient_names)
        self.args = args
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.args.data_type == "feat":
            folder_paths = glob.glob("./data/preprocessed/feat/cross_section_image_cut_repeat/feat/%s_*" % (self.args.data_type, self.patient_names[idx]))
        else:
            folder_paths = glob.glob("./data/preprocessed/img/cross_section_image_cut_repeat/%s_*" % (self.patient_names[idx]))
        
        bag = []
        for folder_path in folder_paths:
            imgs = np.load(folder_path)
            bag.extend(imgs)
            
        bag = np.array(bag)
        bag = bag.transpose((0, 3, 1, 2))
        bag = torch.from_numpy(bag.astype(np.float32))         
        
        table = self.tables[idx]
        label = self.labels[idx]
        
        
        label= torch.tensor(float(label), dtype=torch.float)
        table= torch.from_numpy(table.astype(np.float32))  
        len_list = len(bag)
        return {"bags": bag, "table": table, "label": label, "len_list": len_list, "patient_name": self.patient_names[idx]}


def load_data_bags(args):  # LIMUC
    ######### load data #######
    test_patient_name = np.load('./data/preprocessed/5_fold/%d/test_patient_name.npy' % (args.fold), allow_pickle=True)
    test_table = np.load('./data/preprocessed/5_fold/%d/test_table.npy' % (args.fold), allow_pickle=True)
    test_label = np.load('./data/preprocessed/5_fold/%d/test_label.npy' % (args.fold), allow_pickle=True)
    train_patient_name = np.load('./data/preprocessed/5_fold/%d/train_patient_name.npy' % (args.fold), allow_pickle=True)
    train_table = np.load('./data/preprocessed/5_fold/%d/train_table.npy' % (args.fold), allow_pickle=True)
    train_label = np.load('./data/preprocessed/5_fold/%d/train_label.npy' % (args.fold), allow_pickle=True)
    val_patient_name = np.load('./data/preprocessed/5_fold/%d/val_patient_name.npy' % (args.fold), allow_pickle=True)
    val_table = np.load('./data/preprocessed/5_fold/%d/val_table.npy' % (args.fold), allow_pickle=True)
    val_label = np.load('./data/preprocessed/5_fold/%d/val_label.npy' % (args.fold), allow_pickle=True)

    # sampler
    bag_label_count = np.array([sum(train_label==0), sum(train_label==1)])
    class_weight = 1 / bag_label_count
    sample_weight = [class_weight[train_label[i]] for i in range(len(train_label))]
    sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(train_label), replacement=True)

    train_dataset = Dataset(args=args, patient_names=train_patient_name, tables=train_table, labels=train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    val_dataset = Dataset(args=args, patient_names=val_patient_name, tables=val_table, labels=val_label)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn)  
    test_dataset = Dataset(args=args, patient_names=test_patient_name, tables=test_table, labels=test_label)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=collate_fn)    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    bags = []
    labels = []
    tables = []
    len_list = []
    patient_name_list = []

    for b in batch:
        bags.extend(b["bags"])
        labels.append(b["label"])
        tables.append(b["table"])
        len_list.append(b["len_list"])
        patient_name_list.append(b["patient_name"])

    labels = torch.stack(labels, dim=0)
    len_list = torch.tensor(len_list)
    bags = torch.stack(bags, dim=0)
    tables = torch.stack(tables, dim=0)
    # patient_name_list = torch.tensor(patient_name_list)
    return {"bags": bags, "table": tables, "label": labels, "len_list": len_list, "patient_name":patient_name_list}
