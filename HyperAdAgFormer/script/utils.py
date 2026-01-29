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
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, recall_score, precision_score
import glob
from sklearn.metrics import roc_auc_score, average_precision_score

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

def make_folder(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    if not os.path.exists(args.output_path + "/acc_graph"):
        os.mkdir(args.output_path + "/acc_graph")
    if not os.path.exists(args.output_path + "/cm"):
        os.mkdir(args.output_path + "/cm")
    if not os.path.exists(args.output_path + "/log_dict"):
        os.mkdir(args.output_path + "/log_dict")
    if not os.path.exists(args.output_path + "/loss_graph"):
        os.mkdir(args.output_path + "/loss_graph")
    if not os.path.exists(args.output_path + "/model"):
        os.mkdir(args.output_path + "/model")
    if not os.path.exists(args.output_path + "/test_metrics"):
        os.mkdir(args.output_path + "/test_metrics")
    return

def save_confusion_matrix(cm, path, title=''):
    plt.figure(figsize=(10, 8), dpi=300)
    cm = cm / cm.sum(axis=-1, keepdims=1)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', annot_kws={"size": 40})
    plt.xlabel('pred', fontsize=24)  
    plt.ylabel('GT', fontsize=24) 
    plt.title(title)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def calcurate_metrix(preds, labels):
    preds_label = (preds>=0.5)*1
    acc = (np.array(preds_label) == np.array(labels)).sum() / len(preds)
    f1 = f1_score(labels, preds_label, average="binary")
    recall = recall_score(labels, preds_label, average="binary")
    precision = precision_score(labels, preds_label, average="binary")    
    cm = confusion_matrix(labels, preds_label)
    kap = cohen_kappa_score(labels, preds_label, weights="quadratic")
    auc = roc_auc_score(labels, preds)
    pr_auc = average_precision_score(labels, preds)
    return {"acc": acc, "macro-f1": f1, "kap": kap, "cm": cm, "recall":recall, "precision":precision, "auc":auc, "pr_auc": pr_auc}

    
def make_loss_graph(args, keep_train_loss, keep_valid_loss, path):
    #loss graph save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(keep_train_loss, label = 'train')
    ax.plot(keep_valid_loss, label = 'valid')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("Losses")
    plt.legend()
    fig.savefig(path)
    plt.close() 
    return

def make_bag_acc_graph(args, train_bag_acc, val_bag_acc, test_bag_acc, path):
    #Bag level accuracy save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_bag_acc, label = 'train bag acc')
    ax.plot(val_bag_acc, label = 'valid bag acc')
    ax.plot(test_bag_acc, label = 'test bag acc')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("accuracy")
    plt.legend()
    fig.savefig(path)
    plt.close()
    return


