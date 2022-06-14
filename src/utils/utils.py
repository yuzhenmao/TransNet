import numpy as np
import scipy.sparse as sp
import pandas as pd
import warnings
from sklearn.metrics import f1_score
import torch
import math
import networkx as nx
from tqdm import tqdm
import os
import pdb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import networkx as nx
import torch.nn as nn
import pymetis
import torch.nn.functional as F
import collections
from random import sample
from queue import Queue
import heapq
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from torch.distributions import Categorical
import scipy
import copy


def load_network(file):
    net = sio.loadmat(file)
    X, A, Y = net['attrb'], net['network'], net['group']
    if not isinstance(X, scipy.sparse.lil_matrix):
        X = lil_matrix(X)
    idx = (Y.sum(1) == 1)
    Y = Y[idx]
    A = A[idx,:][:,idx]
    X = X[idx]
    return A, X.todense(), np.argmax(Y, 1)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features

def column_normalize(tens):
    ret = tens - tens.mean(axis=0)
    return ret

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_add_diag=adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj_add_diag)
    return adj_normalized.astype(np.float32) #sp.coo_matrix(adj_unnorm)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

#================================================================
def process_data(G, name):
    dirs = './src/pre_calculated_' + name
    if os.path.exists(dirs):
        pass
    else:
        os.makedirs(dirs)
    adj = G.adj.to_dense().cpu().numpy()
    actual_adj = adj.copy()
    return actual_adj, 0, 0, 0, 0, 0

#================================================================
def viz(svec, tvec, rootdir, tname, epoch, train_loss, dt_loss, domain_loss, linear=False):

    if not os.path.exists(os.path.join(rootdir, tname)):
        os.makedirs(os.path.join(rootdir, tname))

    def discrete_cmap(n, base_cmap=None):
        """Create an N-bin discrete colormap from the specified input map"""
        base = plt.cm.get_cmap(base_cmap)
        return base.from_list(base.name + str(n), base(np.linspace(0, 1, n)), n)

    color = ['red', 'green', 'yellow', 'blue']
    vec = []
    label = []
    legend = []
    tvec_ = tvec.cpu().tolist()
    vec += tvec_
    label += np.zeros(len(tvec_)).tolist()
    legend += ['target']
    for i in range(len(svec)):
        svec_ = svec[i].cpu().tolist()
        vec += svec_
        label += (np.ones(len(svec_)) + i).tolist()
        legend += (['source_' + str(i)])
    
    vec = np.asarray(vec)
    label = np.asarray(label)
    vec_ = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(vec)
    plt.figure(figsize=(10, 10))

    for i, l in enumerate(legend):
        sub_label = np.where(label==i)
        if i == 0:
            alpha = 0.3
        else:
            alpha = 0.1
        plt.scatter(vec_[:,0][sub_label], vec_[:,1][sub_label], alpha=alpha, cmap=discrete_cmap(10, 'jet'), c=color[i], edgecolors='black', label=l)
    # plt.colorbar()
    plt.grid()
    plt.legend()
    plt.title(f' total loss is {train_loss}\n downstream task loss is {dt_loss}\n domain loss is {domain_loss}')
    if linear is False:
        logdir = os.path.join(rootdir, tname, 'latent'+str(epoch)+'.png')
    else:
        logdir = os.path.join(rootdir, tname, 'linear_latent'+str(epoch)+'.png')
    plt.axis('off')
    plt.savefig(logdir)
    plt.close()


# =========================================================
def viz_single(tvec, rootdir, tname, epoch, loss, label, few_shot, adapt=0, sudo_label=None, ass_node=None):

    if not os.path.exists(os.path.join(rootdir, tname)):
        os.makedirs(os.path.join(rootdir, tname))

    def discrete_cmap(n, base_cmap=None):
        """Create an N-bin discrete colormap from the specified input map"""
        base = plt.cm.get_cmap(base_cmap)
        return base.from_list(base.name + str(n), base(np.linspace(0, 1, n)), n)

    color = ['red']
    vec = []
    legend = []
    tvec_ = tvec.cpu().tolist()
    vec += tvec_
    
    vec = np.asarray(vec)
    label = label.cpu().numpy()
    few_shot = few_shot.cpu().numpy()
    vec_ = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(vec)

    if sudo_label is not None:
        sudo_label = sudo_label.cpu().numpy()
        plt.figure(figsize=(10, 10))
        sc = plt.scatter(vec_[:,0], vec_[:,1], alpha=0.7, cmap=discrete_cmap(np.max(sudo_label)-np.min(sudo_label)+1, 'jet'), c=sudo_label, edgecolors='black', vmin = np.min(sudo_label)-.5, vmax = np.max(sudo_label)+.5)
        cax = plt.colorbar(sc, ticks=np.arange(np.min(sudo_label),np.max(sudo_label)+1))
        plt.grid()
        if adapt == 0:
            plt.title(f'acc is {loss}')
            logdir = os.path.join(rootdir, tname, 'sudo_finetune_full_'+str(epoch+1)+'.png')
        elif adapt == 1:
            # plt.title(f'loss is {loss}')
            logdir = os.path.join(rootdir, tname, 'sudo_pretrain_latent_'+str(epoch)+'.png')

        plt.axis('off')
        plt.savefig(logdir)
        plt.close()

    if ass_node is not None:
        ass_node = ass_node.cpu().numpy()
        plt.figure(figsize=(10, 10))
        sub_label = np.where(ass_node==-1)
        other_label = np.where(ass_node!=-1)
        plt.scatter(vec_[:,0][sub_label], vec_[:,1][sub_label], alpha=0.2, c='white', edgecolors='black')
        sc = plt.scatter(vec_[:,0][other_label], vec_[:,1][other_label], alpha=0.9, cmap=discrete_cmap(np.max(label)+1, 'jet'), c=label[other_label], edgecolors='black', vmin = -.5, vmax = np.max(label)+.5)
        cax = plt.colorbar(sc, ticks=np.arange(np.min(label),np.max(label)+1))
        plt.grid()
        if adapt == 0:
            logdir = os.path.join(rootdir, tname, 'finetune_ass'+str(epoch+1)+'.png')
        elif adapt == 1:
            logdir = os.path.join(rootdir, tname, 'pretrain_ass_'+str(epoch)+'.png')
        plt.axis('off')
        plt.savefig(logdir)
        plt.close()

    plt.figure(figsize=(10, 10))
    sub_label = np.where(few_shot==-1)
    other_label = np.where(few_shot!=-1)
    plt.scatter(vec_[:,0][sub_label], vec_[:,1][sub_label], alpha=0.2, c='white', edgecolors='black')
    sc = plt.scatter(vec_[:,0][other_label], vec_[:,1][other_label], alpha=0.9, cmap=discrete_cmap(np.max(label)+1, 'jet'), c=label[other_label], edgecolors='black', vmin = -.5, vmax = np.max(label)+.5)
    cax = plt.colorbar(sc, ticks=np.arange(np.min(label),np.max(label)+1))
    plt.grid()
    if adapt == 0:
        logdir = os.path.join(rootdir, tname, 'finetune_fewshot'+str(epoch+1)+'.png')
    elif adapt == 1:
        logdir = os.path.join(rootdir, tname, 'pretrain_fewshot_'+str(epoch)+'.png')
    plt.axis('off')
    plt.savefig(logdir)
    plt.close()

    plt.figure(figsize=(10, 10))
    sc = plt.scatter(vec_[:,0], vec_[:,1], alpha=0.7, cmap=discrete_cmap(np.max(label)-np.min(label)+1, 'jet'), c=label, edgecolors='black', vmin = np.min(label)-.5, vmax = np.max(label)+.5)
    cax = plt.colorbar(sc, ticks=np.arange(np.min(label),np.max(label)+1))
    plt.grid()
    if adapt == 0:
        plt.title(f'acc is {loss}')
        logdir = os.path.join(rootdir, tname, 'finetune_full_'+str(epoch+1)+'.png')
    elif adapt == 1:
        # plt.title(f'loss is {loss}')
        logdir = os.path.join(rootdir, tname, 'pretrain_latent_'+str(epoch)+'.png')
    plt.axis('off')
    plt.savefig(logdir)
    plt.close()

# =======================================================
class NodeDistance:

    def __init__(self, G, name, nclass=4):
        """
        :param graph: Networkx Graph.
        """
        self.graph = G
        self.nclass = nclass
        self.name = name

    def get_label(self):
        distance = np.zeros([len(self.graph), len(self.graph)])
        dirs = './src/pre_calculated_' + self.name
        if os.path.exists(os.path.join(dirs, "distance.npy")):
            distance = np.load(os.path.join(dirs, "distance.npy"))
        else:
            for i in tqdm(self.graph.nodes()):
                pr = nx.pagerank(self.graph, personalization={i:1})
                dist = np.array(list(pr.values())).astype(float)
                distance[i,:] = dist
            np.save(os.path.join(dirs, "distance.npy"), distance)

        self.same_node = np.diag(distance).min()
        self.distance = distance

        return torch.FloatTensor(distance)

# ============================================================
def mixup_criterion(pred, lam, y, domain_num=2):
    loss = 0
    for i in range(domain_num):
        loss += lam[i] * F.cross_entropy(pred, y[i], reduction="none")
    return loss.mean()

def mixup_hidden_criterion(pred, y_a, y_b, lam):
    return (lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)).mean()

def mixup_hidden(x, alpha=1.0):
    batch_size = x.size()[0]
    lam = torch.FloatTensor(np.random.beta(alpha, alpha, size=batch_size)).cuda()
    lam = lam.view(-1, 1)
    permuted_idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[permuted_idx, :]
    return x_mix, permuted_idx, lam.view(-1)

# =========================================================
def assign_sudo_label(logprobs, labels, device, dicts = None, gcn_labels=None, all_labels=None, few_shot_idx=None, relax=False):
    temp_l = labels.clone().cpu().numpy()
    temp_p = logprobs.detach().clone().cpu().numpy()
    n = temp_l.max() + 1
    m = temp_p.shape[-1]
    few_shot = temp_p.shape[0] // n
    clusters = {i: [] for i in range(n)}
    for idx, label in enumerate(temp_l):
        clusters[label].append(idx)

    if dicts is None:
        means = {}
        dicts = {}
        dicts_inv = {}
        for i in set(temp_l):
            means[i] = temp_p[clusters[i]].max(0)*0
            for j in range(m):
                means[i][j] = np.sum(np.argmax(temp_p[clusters[i]], 1)==j) / np.sqrt(np.sum(temp_l==i))
        
        q_t = Queue(maxsize = n)
        set_s = set()
        for i in range(m):
            set_s.add(i)
        for i in set(temp_l):
            q_t.put(i)

        if relax is False:
            while not q_t.empty():
                if len(set_s) == 0:
                    for i in set(temp_l):
                        means[i] = temp_p[clusters[i]].max(0)*0
                        for j in range(m):
                            means[i][j] = np.sum(np.argmax(temp_p[clusters[i]], 1)==j) / np.sqrt(np.sum(temp_l==i))
                    break
                lll = q_t.get()
                max_i = np.argmax(means[lll])
                max_p = np.max(means[lll])
                if max_i in set_s:
                    set_s.remove(max_i)
                    dicts[lll] = max_i
                    dicts_inv[max_i] = lll
                elif means[dicts_inv[max_i]][max_i] < max_p:
                    q_t.put(dicts_inv[max_i])
                    means[dicts_inv[max_i]][max_i] = np.min(means[dicts_inv[max_i]]) - 1
                    dicts[lll] = max_i
                    dicts_inv[max_i] = lll
                else:
                    q_t.put(lll)
                    means[lll][max_i] = np.min(means[lll]) - 1

        while not q_t.empty():
            lll = int(q_t.get())
            max_i = np.argmax(means[lll])
            dicts[lll] = int(max_i)
    
    new_label = []
    for label in temp_l:
        new_label.append(dicts.get(label, 0))

    return torch.LongTensor(new_label).to(device), dicts
