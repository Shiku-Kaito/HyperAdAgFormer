import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from utils import *

def eval_net(args, model, test_loader):
    fix_seed(args.seed)
    result_dict = {}
    ################## test ###################
    model.eval()
    ins_gt, bag_gt, ins_pred, bag_pred, attenw, bag_list, patient_name_list, bag_len_list = [], [], [], [], [], [], [], []
    feat_list = []
    with torch.no_grad():
        for iteration, data in enumerate(tqdm(test_loader)): #enumerate(tqdm(test_loader, leave=False)):
            bags, table, label = data["bags"], data["table"], data["label"]
            bags, table, label  = bags.to(args.device), table.to(args.device), label.to(args.device)

            y = model(bags, table, data["len_list"])
            
            bag_gt.extend(label.cpu().detach().numpy())
            bag_pred.extend(y["y_bag"].cpu().detach().numpy())
            attenw.extend(y["atten_weight"])
            bag_list.extend(bags.cpu().detach().numpy())
            patient_name_list.extend(data["patient_name"])
            bag_len_list.extend(data["len_list"])
            feat_list.extend(y["bag_feat"].cpu().detach().numpy())


    bag_gt, bag_pred =np.array(bag_gt), np.array(bag_pred)
    bag_metric = calcurate_metrix(bag_pred, bag_gt)
    test_bag_cm = bag_metric["cm"]

    result_dict["bag_acc"], result_dict["bag_macro-f1"], result_dict["bag_auc"], result_dict["bag_pr_auc"] = bag_metric["acc"], bag_metric["macro-f1"], bag_metric["auc"], bag_metric["pr_auc"]
    result_dict["recall"], result_dict["precision"] = bag_metric["recall"], bag_metric["precision"]
    result_dict["bag_pred"], result_dict["bag_gt"] = bag_pred, bag_gt
    result_dict["attenw"] = attenw
    result_dict["bag_list"] = bag_list
    result_dict["patient_name_list"] = patient_name_list
    result_dict["bag_len_list"] = bag_len_list
    result_dict["feats"] = np.array(feat_list)
    return result_dict