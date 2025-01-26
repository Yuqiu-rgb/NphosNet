# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import sklearn
import torch
from read_data import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score, recall_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from NphosNet import *
from train import BERT_encoding

import pickle
import config
import argparse

if __name__ == "__main__":
    cf = config.get_train_config()
    cf.task = 'test'
    # device = torch.device(cf.device if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    BERT_model = BERT(cf)

    '''ph'''
    path = '../model/PH_train'
    train, test = data_read_ph()
    train_seq = train.iloc[:, 1]
    test_seq = test.iloc[:, 1]
    test_label = torch.tensor(np.array(test.iloc[:, 0], dtype='int64')).to(device,
                                                                               non_blocking=True)  # Very important
    train_encoding = BERT_encoding(train_seq, test_seq).to(device, non_blocking=True)
    test_encoding = BERT_encoding(test_seq, train_seq).to(device, non_blocking=True)
    test_embedding = torch.tensor(np.load('../embedding/PH_test_embedding.npy')).to(device,
                                                                                       non_blocking=True)
    test_str_embedding = torch.tensor(np.load('../embedding/PH_test_str_embedding.npy')).to(
            device, non_blocking=True)

    '''pk'''
    # path = '../model/PK_train'
    # train, test = data_read_pk()
    # train_seq = train.iloc[:, 1]
    # test_seq = test.iloc[:, 1]
    # test_label = torch.tensor(np.array(test.iloc[:, 0], dtype='int64')).to(device,
    #                                                                            non_blocking=True)  # Very important
    # train_encoding = BERT_encoding(train_seq, test_seq).to(device, non_blocking=True)
    # test_encoding = BERT_encoding(test_seq, train_seq).to(device, non_blocking=True)
    # test_embedding = torch.tensor(np.load('../embedding/PK_test_embedding.npy')).to(device,
    #                                                                                    non_blocking=True)
    # test_str_embedding = torch.tensor(np.load('../embedding/PK_test_str_embedding.npy')).to(
    #         device, non_blocking=True)

    '''PR'''
    # path = '../model/PR_train'
    # train, test = data_read_pr()
    # train_seq = train.iloc[:, 1]
    # test_seq = test.iloc[:, 1]
    # test_label = torch.tensor(np.array(test.iloc[:, 0], dtype='int64')).to(device,
    #                                                                            non_blocking=True)  # Very important
    # train_encoding = BERT_encoding(train_seq, test_seq).to(device, non_blocking=True)
    # test_encoding = BERT_encoding(test_seq, train_seq).to(device, non_blocking=True)
    # test_embedding = torch.tensor(np.load('../embedding/PR_test_embedding.npy')).to(device,
    #                                                                                    non_blocking=True)
    # test_str_embedding = torch.tensor(np.load('../embedding/PR_test_str_embedding.npy')).to(
    #     device, non_blocking=True)


    # Get all file names in the directory
    files = os.listdir(path)
    # Filter out file names ending with '.pt'
    # pt_files = [os.path.join(path, f) for f in files if f.endswith('.pt')]
    pt_files = [os.path.join(path, f) for f in files if f.endswith('fold1_BERT_model.pt')]

    # Evaluate all .pt models
    BERT_test_auc = 0
    num = 0
    Result = []
    Result_softmax = []

    for f in pt_files:
        print('loading model ', f)
        BERT_model.load_state_dict(torch.load(f))
        BERT_model = BERT_model.to(device, non_blocking=True)
        BERT_model.eval()

        # Use the following line for models without the structure embedding
        # result, _ = BERT_model(test_encoding,test_embedding)
        # Use the following line for models with the structure embedding
        result, _ = BERT_model(test_encoding, test_embedding, test_str_embedding)
        result_softmax = F.softmax(result, dim=1)  # Apply softmax to the output

        Result.append(result)
        Result_softmax.append(result_softmax)

        _, predicted = torch.max(result_softmax, 1)
        correct = (predicted == test_label).sum().item()
        result = result.cpu().detach().numpy()
        result_softmax = result_softmax.cpu().detach().numpy()
        BERT_test_acc, BERT_test_prob = 100 * correct / result.shape[0], result_softmax

        # Calculate True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN)
        tn, fp, fn, tp = confusion_matrix(test_label.cpu(), predicted.cpu()).ravel()
        # Calculate accuracy (ACC)
        BERT_test_acc = 100 * (tp + tn) / (tp + tn + fp + fn)
        # Calculate sensitivity (SEN)
        BERT_sen = tp / (tp + fn)
        # Calculate specificity (SPEC)
        BERT_spec = tn / (tn + fp)
        # Calculate Matthew's Correlation Coefficient (MCC)
        BERT_mcc = matthews_corrcoef(test_label.cpu(), predicted.cpu())
        # Calculate AUC
        BERT_test_auc = roc_auc_score(test_label.cpu(), BERT_test_prob[:, 1])
        precision, recall, thresholds = precision_recall_curve(test_label.cpu(),
                                                               BERT_test_prob[:, 1])

        # 计算平均精确度（AP）
        average_precision = average_precision_score(test_label.cpu(), BERT_test_prob[:, 1])

        result_str = 'Model file name: {}  Accuracy: {:.4f} SEN: {:.4f} SPEC: {:.4f} MCC: {:.4f} AUC: {:.4f} AP: {:.4f}\n'.format(
            f, BERT_test_acc, BERT_sen, BERT_spec, BERT_mcc, BERT_test_auc,average_precision)
        print(result_str)
        # print(f'tn:{tn},fp:{fp},fn:{fn},tp:{tp}')

        with open(os.path.join(path, 'PTransIPs_text_result.txt'), "a") as f:
            f.write(result_str)

    # Compute mean of Result and Result_softmax
    mean_Result_softmax = np.mean([t.cpu().detach().numpy() for t in Result_softmax], axis=0)
    mean_Result_softmax = torch.tensor(mean_Result_softmax)
    mean_Result = np.mean([t.cpu().detach().numpy() for t in Result], axis=0)

    # Convert predictions from each model to binary labels and vote
    votes = [torch.argmax(t, dim=1) for t in Result_softmax]
    votes = torch.stack(votes)
    votes_sum = torch.sum(votes, dim=0)

    predicted = torch.where(votes_sum > len(votes) / 2, torch.ones_like(votes_sum),
                            torch.zeros_like(votes_sum))

    # Calculate True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN)
    tn, fp, fn, tp = confusion_matrix(test_label.cpu(), predicted.cpu()).ravel()
    # Calculate accuracy (ACC)
    BERT_test_acc = 100 * (tp + tn) / (tp + tn + fp + fn)
    # Calculate sensitivity (SEN)
    BERT_sen = tp / (tp + fn)
    # Calculate specificity (SPEC)
    BERT_spec = tn / (tn + fp)
    # Calculate Matthew's Correlation Coefficient (MCC)
    BERT_mcc = matthews_corrcoef(test_label.cpu(), predicted.cpu())
    # Calculate AUC
    BERT_test_auc = roc_auc_score(test_label.cpu(), BERT_test_prob[:, 1])

    result_str = 'All kfold Model file:  Accuracy: {:.4f} SEN: {:.4f} SPEC: {:.4f} MCC: {:.4f} AUC: {:.4f}\n'.format(
        BERT_test_acc, BERT_sen, BERT_spec, BERT_mcc, BERT_test_auc)
    print(result_str)
    # print(f'tn:{tn},fp:{fp},fn:{fn},tp:{tp}')

    with open(os.path.join(path, 'PTransIPs_text_result.txt'), "a") as f:
        f.write(result_str)
    fpr, tpr, thresholds = roc_curve(test_label.cpu(), BERT_test_prob[:, 1])

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % BERT_test_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    # plt.savefig('../model/PH_train/ph_auc.png')
    # plt.savefig('../model/PK_train/pk_auc.png')
    # plt.savefig('../model/PR_train/pr_auc.png')

    fpr1 = list(fpr)
    tpr1 = list(tpr)
    precision, recall, thresholds = precision_recall_curve(test_label.cpu(), BERT_test_prob[:, 1])

    #计算平均精确度（AP）
    average_precision = average_precision_score(test_label.cpu(), BERT_test_prob[:, 1])
    print(f'AP:{average_precision}')
    recall1 = list(recall)
    precision1 = list(precision)
    # 绘制AP曲线
    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    # plt.plot([0, 1], [0.5, 0.5], 'k--', lw=2)  # 绘制一个基线
    # plt.savefig('../model/PH_train/ph_ap.png')
    # plt.savefig('../model/PK_train/pk_ap.png')
    # plt.savefig('../model/PR_train/pr_ap.png')


    np.save(os.path.join(path, 'PTransIPs_test_prob.npy'), BERT_test_prob)
