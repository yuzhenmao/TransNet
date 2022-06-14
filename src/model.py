#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from gnns import GNN
import numpy as np
import random
from random import sample
from utils.utils import NodeDistance, mixup_hidden, mixup_hidden_criterion, assign_sudo_label
from multiprocessing import Pool
import networkx as nx
import collections
import pdb


logger = logging.getLogger(__name__)


class GradientReversalLayer(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(ctx, inputs):
        # ctx.save_for_backward(inputs)
        return inputs                             

    @staticmethod
    def backward(ctx, grad_output):
        # result, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = -grad_input * ratio

        return grad_input


class TransNet(nn.Module):
    """
    Multi-layer perceptron with adversarial regularizer by domain classification.
    """
    def __init__(self, configs):
        super(TransNet, self).__init__()
        # self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        # self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        self.num_domains = configs["num_domains"]
        self.num_classes = configs["num_classes"]
        # Parameters of hidden, fully-connected layers, feature learning component.
        # self.h_dim = min(configs["input_dim"])
        self.h_dim = configs['feat_num']
        self.hiddens = GNN(nfeat=self.h_dim,
                              nhid=configs["hidden_layers"],
                              nclass=2,    # not used
                              ndim=configs["ndim"],
                              gnn_type=configs["type"],
                              bias=True,
                              dropout=configs["dropout"])
        self.domain_disc = torch.nn.Sequential(
            nn.Linear(configs["ndim"], configs["ndim"]),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(configs["ndim"], 2),
            # nn.Linear(configs["ndim"], 2),
        )
        self.domain_disc_linear = torch.nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(self.h_dim, 2),
            # nn.Linear(configs["ndim"], 2),
        )
        # Parameter of the first dimension reduction layer.
        self.dimRedu = nn.ModuleList([torch.nn.Sequential(nn.Dropout(p=0.7), nn.Linear(ndim, self.h_dim)) for ndim in configs["input_dim"]])
        # Parameter of the final softmax classification layer.
        self.triplet_embedding = nn.ModuleList([torch.nn.Sequential(
                                            # nn.Dropout(p=0.5),
                                            nn.Linear(2*configs["ndim"], configs["ndim"]),
                                            # nn.ReLU(),
                                            ) for _ in configs["num_classes"]])
        self.classifier = nn.ModuleList([torch.nn.Sequential(
                                            nn.Dropout(p=0.5),
                                            # nn.Linear(2*configs["ndim"], 2*configs["ndim"]),
                                            # nn.ReLU(),
                                            nn.Linear(configs["ndim"], 2*nclass+1)) for nclass in configs["num_classes"]])
        self.gnn_classifier = GNN(nfeat=configs["ndim"], 
                                nhid=[configs["ndim"]], 
                                nclass=2, 
                                ndim=configs["num_classes"][-1],
                                gnn_type=configs["type"], 
                                bias=True, 
                                dropout=0.5, 
                                batch_norm=False)
        # Parameter of the domain classification layer, multiple sources single target domain adaptation.
        self.domains = nn.ModuleList([self.domain_disc for _ in range(self.num_domains)])
        self.domains_linear = nn.ModuleList([self.domain_disc_linear for _ in range(self.num_domains)])
        # Gradient reversal layer.
        self.grls = [GradientReversalLayer.apply for _ in range(self.num_domains)]
        self.t_class = configs["num_classes"][-1]
        self.t_dim = configs["ndim"]

        # TODO make the domain discriminator more complicated


    def forward(self, sinputs, tinputs, sadj, tadj, rate):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:
        """
        global ratio
        ratio = rate

        sh_relu = []
        sh_linear = []
        th_relu = tinputs.clone()
        for i in range(self.num_domains):
            sh_relu.append(sinputs[i].clone())

        for i in range(self.num_domains):
            sh_linear.append(self.dimRedu[i](sh_relu[i]))
            sh_relu[i] = F.relu(self.hiddens(sh_linear[i], sadj[i]))  # check the output of dimRedu (TSNE)
        th_linear = self.dimRedu[-1](th_relu)
        th_relu = F.relu(self.hiddens(th_linear, tadj))
        # Domain classification accuracies.
        sdomains, tdomains, sdomains_linear, tdomains_linear, mixups = [], [], [], [], []
        for i in range(self.num_domains):
            sdomains.append(F.log_softmax(self.domains[i](self.grls[i](sh_relu[i])), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.grls[i](th_relu)), dim=1))
            sdomains_linear.append(F.log_softmax(self.domains_linear[i](self.grls[i](sh_linear[i])), dim=1))
            tdomains_linear.append(F.log_softmax(self.domains_linear[i](self.grls[i](th_linear)), dim=1))

        return sh_relu, th_relu, sh_linear, th_linear, sdomains, tdomains, sdomains_linear, tdomains_linear

    def finetune(self):
        for param in self.hiddens.parameters():
            param.requires_grad = False
        for param in self.dimRedu[-1].parameters():
            param.requires_grad = False

    def finetune_inv(self):
        for param in self.hiddens.parameters():
            param.requires_grad = True
        for param in self.dimRedu[-1].parameters():
            param.requires_grad = True

    def inference(self, tinput, adj, index=-1, sudo=False):
        h_relu = tinput.clone()
        h_linear = self.dimRedu[index](h_relu)
        h_relu = F.relu(self.hiddens(h_linear, adj))
        # Classification probability.
        if sudo is True:
            index = 0
        if index == -1:
            logprobs = F.log_softmax(self.gnn_classifier(h_relu, adj), dim=1)
        else:
            logprobs = F.log_softmax(self.classifier[index](self.triplet_embedding[index](torch.cat((h_relu, h_relu), 1)))[:, :self.num_classes[index]], dim=1)
        return logprobs, h_relu, h_linear


# ======================================================
class GeneralSignal(object):

    def __init__(self, graph, adj, labels, features, nhid, device, dataset, args, seed, model, index, _lambda, dicts=None):
        self.G = graph.G
        self.model = model
        self.node_num = len(self.G)
        self.name = graph.name
        self.adj = adj.clone()
        self.adj = self.adj.to_dense().cpu().numpy()
        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device
        self.sample_num = []
        self.train_sample_num = []
        self.vali_sample_num = []
        self.test_sample_num = []
        self.labels = labels
        self.nlabel = self.labels.max()+1
        self.index=index
        self.idx_train = np.sort(np.array(dataset.train_indices))
        self.idx_test = np.sort(np.array(dataset.test_indices))
        self.idx_vali = np.sort(np.array(dataset.val_indices))
        self.idx_finetune = np.sort(np.array(dataset.finetune_indices))
        self._lambda = _lambda
        self.alpha = args.alpha
        self.mixup = args.no_mixup
        self.dicts = dicts
        
        random.seed(seed)
        np.random.seed(seed)

        self.all = np.arange(self.adj.shape[0])
        self.unlabeled = np.array([n for n in self.all if n not in self.idx_train])

        d = {'pubmed': 30, 'cora':10, 'citeseer': 10, 'reddit1401_graph1': 10, 'reddit1401_graph2': 10, 'reddit1401_graph3': 10}  # Maybe set the actual label number?
        self.nclass = 6

        self.agent = NodeDistance(self.G, nclass=self.nclass, name=self.name)
        self.pseudo_labels = self.agent.get_label().to(self.device)
        self.train_pseudo_labels = self.pseudo_labels[self.idx_train,:][:,self.idx_train]
        self.train_distance = self.agent.distance[self.idx_train,:][:,self.idx_train]

        self.train_tmps = []
        self.tmps = []

        ###############################################################
        self.thresh = [self.agent.distance.max()+1, self.agent.same_node, 0.01, 0.004, 0.001, 0.0002, 0]

        for i in range(0, 6): 
            self.tmps.append(np.array(np.where((self.agent.distance < self.thresh[i])&(self.agent.distance >= self.thresh[i+1]))).transpose())
            if i == 0:
                self.sample_num.append(self.tmps[0].shape[0] // 10)
            else:
                self.sample_num.append(self.sample_num[0]*2*i)

        for i in range(0, 6):
            self.train_tmps.append(np.array(np.where((self.train_distance < self.thresh[i])&(self.train_distance >= self.thresh[i+1]))).transpose())
            if i == 0:
                self.train_sample_num.append(self.train_tmps[0].shape[0] // 10)
            else:
                self.train_sample_num.append(self.train_sample_num[0]*2*i)

        print(self.train_sample_num)

    def get_dicts(self, embeddings, idx_assis, labels, relax=False):
        label0_loss, label1_loss, dist_loss = 0.0, 0.0, 0.0
        permu = np.random.permutation(idx_assis.shape[0])
        few_shot_pair = torch.cat((embeddings[idx_assis], embeddings[idx_assis][permu]), 1)
        few_shot_embedding = self.model.triplet_embedding[0](few_shot_pair)
        few_shot_output = self.model.classifier[0](few_shot_embedding)
        sudo_label_num = (few_shot_output.shape[-1]-1)//2
        few_shot_pred_label0 = F.log_softmax(few_shot_output[:, :sudo_label_num], dim=1)
        few_shot_pred_label1 = F.log_softmax(few_shot_output[:, sudo_label_num:2*sudo_label_num], dim=1)
        few_shot_pred_label = torch.cat((few_shot_pred_label0, few_shot_pred_label1), 0)
        few_shot_label = torch.cat((labels, labels[permu]), 0)
        sudo_labels, self.dicts = assign_sudo_label(few_shot_pred_label, few_shot_label, self.device, relax=relax)


    def make_loss(self, embeddings, dicts=None, task='train'):
        if self.index == -1:
            node_pairs = self.sample(self.agent.distance, self.tmps, k=self.sample_num)
            embeddings0 = embeddings[node_pairs[0]]
            embeddings1 = embeddings[node_pairs[1]]
        else:
            if task == 'train':
                node_pairs = self.sample(self.train_distance, self.train_tmps, k=self.train_sample_num)
                # node_pairs = [np.arange(self.idx_train.shape[0]), np.arange(self.idx_train.shape[0])]
                embeddings0 = embeddings[self.idx_train[node_pairs[0]]]
                embeddings1 = embeddings[self.idx_train[node_pairs[1]]]
            elif task == 'vali':
                # node_pairs = self.sample(self.vali_agent.distance, self.vali_tmps, k=self.vali_sample_num)
                node_pairs = [self.idx_vali, self.idx_vali]
                embeddings0 = embeddings[node_pairs[0]]
                embeddings1 = embeddings[node_pairs[1]]

        embedding_pair = torch.cat((embeddings0, embeddings1), 1)

        hidden_embedding = self.model.triplet_embedding[self.index](embedding_pair)
        output = self.model.classifier[self.index](hidden_embedding)

        label0_loss, label1_loss, dist_loss = 0.0, 0.0, 0.0
        if self.index == -1:  # TODO, add sampled distance in the future
            permu = np.random.permutation(self.idx_finetune.shape[0])
            few_shot_pair = torch.cat((embeddings[self.idx_finetune], embeddings[self.idx_finetune][permu]), 1)
            few_shot_embedding = self.model.triplet_embedding[0](few_shot_pair)
            few_shot_output = self.model.classifier[0](few_shot_embedding)
            sudo_label_num = (few_shot_output.shape[-1]-1)//2
            few_shot_pred_label0 = F.log_softmax(few_shot_output[:, :sudo_label_num], dim=1)
            few_shot_pred_label1 = F.log_softmax(few_shot_output[:, sudo_label_num:2*sudo_label_num], dim=1)
            few_shot_pred_label = torch.cat((few_shot_pred_label0, few_shot_pred_label1), 0)
            few_shot_label = torch.cat((self.labels[self.idx_finetune], self.labels[self.idx_finetune][permu]), 0)
            sudo_labels, _ = assign_sudo_label(few_shot_pred_label, few_shot_label, self.device, dicts=dicts)
            pred_dist = output[:, -1]
            dist_loss_p = F.mse_loss(pred_dist, self.pseudo_labels[node_pairs], reduction='mean')
            if self.mixup:
                iter_num = 1
                for _ in range(iter_num):
                    x_mix, permuted_idx, lam = mixup_hidden(few_shot_embedding, self.alpha)
                    o_mix = self.model.classifier[0](x_mix)
                    p_mix0 = o_mix[:, :sudo_label_num]
                    p_mix1 = o_mix[:, sudo_label_num:2*sudo_label_num]
                    label0_loss += mixup_hidden_criterion(p_mix0, sudo_labels[:self.idx_finetune.shape[0]], sudo_labels[:self.idx_finetune.shape[0]][permuted_idx], lam)
                    label1_loss += mixup_hidden_criterion(p_mix1, sudo_labels[self.idx_finetune.shape[0]:], sudo_labels[self.idx_finetune.shape[0]:][permuted_idx], lam)
                    x_mix, permuted_idx, lam = mixup_hidden(hidden_embedding, self.alpha)
                    o_mix = self.model.classifier[0](x_mix)
                    pd_mix = o_mix[:, -1]
                    d_mix = lam * self.pseudo_labels[node_pairs] + (1 - lam) * self.pseudo_labels[node_pairs][permuted_idx]
                    dist_loss += ((dist_loss_p + F.mse_loss(pd_mix, d_mix, reduction='mean')) / 2)
                label0_loss, label1_loss, dist_loss = label0_loss/iter_num, label1_loss/iter_num, dist_loss/iter_num
            else:
                label0_loss = F.nll_loss(few_shot_pred_label0, sudo_labels[:self.idx_finetune.shape[0]])
                label1_loss = F.nll_loss(few_shot_pred_label1, sudo_labels[self.idx_finetune.shape[0]:])
                dist_loss = dist_loss_p
        else:
            if task == 'train':
                task_labels = self.train_pseudo_labels
                task_index = self.idx_train
            elif task == 'vali':
                task_labels = self.pseudo_labels
                task_index = self.all
            # node_samples = list(set(node_pairs[1]) | set(node_pairs[0]))
            pred_label0 = output[:, :self.nlabel]
            pred_label1 = output[:, self.nlabel:2*self.nlabel]
            pred_label0 = F.log_softmax(pred_label0, dim=1)
            pred_label1 = F.log_softmax(pred_label1, dim=1)
            label0_loss = F.nll_loss(pred_label0, self.labels[task_index[node_pairs[0]]])
            label1_loss = F.nll_loss(pred_label1, self.labels[task_index[node_pairs[1]]])
            pred_dist = output[:, -1]
            dist_loss = F.mse_loss(pred_dist, task_labels[node_pairs], reduction='mean')
            # Mixup for the hidden layer
            if task == 'train':
                if self.mixup:
                    x_mix, permuted_idx, lam = mixup_hidden(hidden_embedding, self.alpha)
                    o_mix = self.model.classifier[self.index](x_mix)
                    p_mix0 = o_mix[:, :self.nlabel]
                    p_mix1 = o_mix[:, self.nlabel:2*self.nlabel]
                    label0_loss = mixup_hidden_criterion(p_mix0, self.labels[task_index[node_pairs[0]]], self.labels[task_index[node_pairs[0]]][permuted_idx], lam)
                    label1_loss = mixup_hidden_criterion(p_mix1, self.labels[task_index[node_pairs[1]]], self.labels[task_index[node_pairs[1]]][permuted_idx], lam)
                    pd_mix = o_mix[:, -1]
                    d_mix = lam * task_labels[node_pairs] + (1 - lam) * task_labels[node_pairs][permuted_idx]
                    dist_loss = (dist_loss + F.mse_loss(pd_mix, d_mix, reduction='mean')) / 2      

        return (label0_loss+label1_loss)/2 + self._lambda*dist_loss

    def sample(self, labels, tmps, k=[500, 1000, 2000, 4000, 6000, 8000]):
        node_pairs = []
        for i in range(0, 6):
            tmp = tmps[i]
            indices = sample(range(len(tmp)), min(k[i], len(tmp)))
            node_pairs.append(np.array(tmp[indices]))
        node_pairs = np.concatenate((node_pairs[0], node_pairs[1], node_pairs[2], node_pairs[3], node_pairs[4], node_pairs[5]), 0).reshape(-1, 2).transpose()

        return node_pairs[0], node_pairs[1]
