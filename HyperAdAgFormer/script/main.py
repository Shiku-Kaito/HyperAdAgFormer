import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from utils import *
from get_module import get_module
import torch.nn.functional as F

def main(args):
    fix_seed(args.seed) 
    train_net, eval_net, model, optimizer, loss_function, train_loader, val_loader, test_loader = get_module(args)
    args.output_path += '/%s/w_pretrain_lr=3e-6/%s/' % (args.data_type, args.mode) 
    make_folder(args)

    if args.is_evaluation == False:
        train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function)
        return
    else:    
        model.load_state_dict(torch.load(("%s/model/fold=%d_seed=%d-best_model.pkl") % (args.output_path, args.fold, args.seed) ,map_location=args.device))
        result_dict = eval_net(args, model, test_loader)   
    return result_dict


if __name__ == '__main__':
    results_dict = {"bag_acc":[], "bag_kap":[], "bag_macro-f1":[], "recall":[], "precision":[], "auc":[], "bag_pr_auc":[]}
    
    for fold in range(5):
    #    for seed in range(1):
        parser = argparse.ArgumentParser()
        # Data selectiion
        parser.add_argument('--fold', default=fold,
                            type=int, help='fold number')
        parser.add_argument('--classes', #書き換え
                            default=2, type=int, help="number of the sampled instnace")
        # Save Path
        parser.add_argument('--output_path',
                            default='./result_github/', type=str, help="output file name")
        # Training Setup
        parser.add_argument('--num_epochs', default=2000, type=int,
                            help='number of epochs for training.')
        parser.add_argument('--patience', default=50,
                            type=int, help='patience of early stopping')
        parser.add_argument('--device', default='cuda:0',
                            type=str, help='device')
        parser.add_argument('--batch_size', default=16,
                            type=int, help='batch size for training.')
        parser.add_argument('--seed', default=0,
                            type=int, help='seed value')
        parser.add_argument('--num_workers', default=0, type=int,
                            help='number of workers for training.')
        parser.add_argument('--lr', default=3e-6,
                            type=float, help='learning rate')        
        parser.add_argument('--is_evaluation', default=0,
                            type=int, help='1 or 0')                       
        parser.add_argument('--data_type', default="img",
                            type=str, help='img or feat')              
        # Module Selection
        parser.add_argument('--module',default='HyperAdAgFormer', 
                            type=str, help="HyperAdAgFormer")
        parser.add_argument('--mode',default='',    # don't write!
                            type=str, help="")                        
        ### Module detail ####
        parser.add_argument('--transfomer_layer_num', default=1, type=int, help='1 or 2 or 6 or 12')
        
        args = parser.parse_args()

        # train
        if args.is_evaluation == False:
            main(args)
        # evalation
        else:
            result_dict = main(args)
            results_dict["bag_acc"].append(result_dict["bag_acc"]), results_dict["bag_macro-f1"].append(result_dict["bag_macro-f1"]), results_dict["auc"].append(result_dict["bag_auc"]), results_dict["bag_pr_auc"].append(result_dict["bag_pr_auc"])
            results_dict["recall"].append(result_dict["recall"]), results_dict["precision"].append(result_dict["precision"])

            print("############ fold=%d ###########" % args.fold)
            print("@ Bag acc:%.5f, recall:%.5f, precision:%.5f, macro-f1:%.5f, AUC:%.5f, PR-AUC:%.5f" % (float(result_dict["bag_acc"]), float(result_dict["recall"]), float(result_dict["precision"]), float(result_dict["bag_macro-f1"]), float(result_dict["bag_auc"]), float(result_dict["bag_pr_auc"])))
                 
    if args.is_evaluation == True:
        print("5-fold cross-validation")
        print("@Bag acc:%.3f$\pm$%.3f, recall:%.3f$\pm$%.3f, precision:%.3f$\pm$%.3f, macro-f1:%.3f$\pm$%.3f, auc:%.3f$\pm$%.3f, PR-AUC:%.3f$\pm$%.3f" % ((np.array(results_dict["bag_acc"]).mean()), np.std(np.array(results_dict["bag_acc"])), (np.array(results_dict["recall"]).mean()), np.std(np.array(results_dict["recall"])), (np.array(results_dict["precision"]).mean()), np.std(np.array(results_dict["precision"])), (np.array(results_dict["bag_macro-f1"]).mean()), np.std(np.array(results_dict["bag_macro-f1"])), (np.array(results_dict["auc"]).mean()), np.std(np.array(results_dict["auc"])), (np.array(results_dict["bag_pr_auc"]).mean()), np.std(np.array(results_dict["bag_pr_auc"]))))
        
        