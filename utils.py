import copy
import json
import math
import pickle
import random
import re
from collections import Counter

import nltk
import torch
import numpy as np
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn.utils.rnn import pad_sequence

torch.autograd.set_detect_anomaly(True)


def pad_tensors(tensors):
    """
    Takes a list of `N` M-dimensional tensors (M<4) and returns a padded tensor.

    The padded tensor is `M+1` dimensional with size `N, S1, S2, ..., SM`
    where `Si` is the maximum value of dimension `i` amongst all tensors.
    """
    rep = tensors[0]
    padded_dim = []
    for dim in range(rep.dim()):
        max_dim = max([tensor.size(dim) for tensor in tensors])
        padded_dim.append(max_dim)
    padded_dim = [len(tensors)] + padded_dim
    padded_tensor = torch.zeros(padded_dim)
    padded_tensor = padded_tensor.type_as(rep)
    for i, tensor in enumerate(tensors):
        size = list(tensor.size())
        if len(size) == 1:
            padded_tensor[i, :size[0]] = tensor
        elif len(size) == 2:
            padded_tensor[i, :size[0], :size[1]] = tensor
        elif len(size) == 3:
            padded_tensor[i, :size[0], :size[1], :size[2]] = tensor
        else:
            raise ValueError('Padding is supported for upto 3D tensors at max.')
    return padded_tensor


def ints_to_tensor(ints):
    """
    Converts a nested list of integers to a padded tensor.
    """
    if isinstance(ints, torch.Tensor):
        return ints
    if isinstance(ints, list):
        if isinstance(ints[0], int):
            return torch.LongTensor(ints)
        if isinstance(ints[0], torch.Tensor):
            return pad_tensors(ints)
        if isinstance(ints[0], list):
            return ints_to_tensor([ints_to_tensor(inti) for inti in ints])

def get_node_mask(edu_nums, node_num):
    node_list = []
    for index,edu_num in enumerate(edu_nums):
        temp =  [1]*edu_num+[0]*(node_num-edu_num)
        node_list.append(temp)
    return torch.LongTensor(np.array(node_list,dtype=np.long)).cuda()


def get_mask(node_num, max_edu_dist):
    batch_size, max_num=node_num.size(0), node_num.max()
    mask=torch.arange(max_num).unsqueeze(0).cuda()<node_num.unsqueeze(1)
    mask=mask.unsqueeze(1).expand(batch_size, max_num, max_num)
    mask=mask&mask.transpose(1,2)
    mask = torch.tril(mask, -1)
    if max_num > max_edu_dist:
        mask = torch.triu(mask, max_edu_dist - max_num)
    return mask


def compute_loss(link_scores, label_scores, graphs, mask, p=False, negative=False):
    link_scores[~mask]=-1e9
    label_mask=(graphs!=0)&mask
    tmp_mask=(graphs.sum(-1)==0)&mask[:,:,0]
    link_mask=label_mask.clone()
    link_mask[:,:,0]=tmp_mask
    link_scores=torch.nn.functional.softmax(link_scores, dim=-1)
    link_loss=-torch.log(link_scores[link_mask])
    vocab_size=label_scores.size(-1)
    label_loss=torch.nn.functional.cross_entropy(label_scores[label_mask].reshape(-1, vocab_size), graphs[label_mask].reshape(-1), reduction='none')
    if negative:
        negative_mask=(graphs==0)&mask
        negative_loss=torch.nn.functional.cross_entropy(label_scores[negative_mask].reshape(-1, vocab_size), graphs[negative_mask].reshape(-1),reduction='mean')
        return link_loss, label_loss, negative_loss
    if p:
        return link_loss, label_loss, torch.nn.functional.softmax(label_scores[label_mask],dim=-1)[torch.arange(label_scores[label_mask].size(0)),graphs[mask]]
    return link_loss, label_loss



def record_eval_result(eval_matrix, predicted_result):
    for k, v in eval_matrix.items():
        if v is None:
            if isinstance(predicted_result[k], dict):
                eval_matrix[k] = [predicted_result[k]]
            else:
                eval_matrix[k] = predicted_result[k]
        elif isinstance(v, list):
            eval_matrix[k] += [predicted_result[k]]
        else:
            eval_matrix[k] = np.append(eval_matrix[k], predicted_result[k])


def tsinghua_F1(eval_matrix):
    cnt_golden, cnt_pred, cnt_cor_bi, cnt_cor_multi = 0, 0, 0, 0
    for hypothesis, reference, edu_num in zip(eval_matrix['hypothesis'], eval_matrix['reference'],
                                              eval_matrix['edu_num']):
        cnt = [0] * edu_num
        for r in reference:
            cnt[r[1]] += 1
        for i in range(edu_num):
            if cnt[i] == 0:
                cnt_golden += 1
        cnt_pred += 1
        if cnt[0] == 0:
            cnt_cor_bi += 1
            cnt_cor_multi += 1
        cnt_golden += len(reference)
        cnt_pred += len(hypothesis)
        for pair in hypothesis:
            if pair in reference:
                cnt_cor_bi += 1
                if hypothesis[pair] == reference[pair]:
                    cnt_cor_multi += 1
    prec_bi, recall_bi = cnt_cor_bi * 1. / cnt_pred, cnt_cor_bi * 1. / cnt_golden
    f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    prec_multi, recall_multi = cnt_cor_multi * 1. / cnt_pred, cnt_cor_multi * 1. / cnt_golden
    f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    return f1_bi, f1_multi
