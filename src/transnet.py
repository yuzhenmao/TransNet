#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import argparse
import pickle
import torch
import torch.optim as optim
import networkx as nx
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from model import TransNet, GeneralSignal
import matplotlib.pyplot as plt
from base_model import BASE
from util import get_logger
from util import data_loader, multi_data_loader
from utils.utils import process_data, viz, viz_single, mixup_criterion, assign_sudo_label
from utils.dataloader import GraphLoader, MyDataset, MyData
from torch.utils.data import DataLoader
from utils.logger import Logger
from sklearn.metrics import f1_score, confusion_matrix
import warnings
import os
import json
from tqdm import tqdm
import sys
import pdb
import copy


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="amazon")
parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=93)
parser.add_argument("-u", "--mu", help="coefficient for the domain adversarial loss", type=float, default=1e-2)
parser.add_argument('--hidden', type=list, default=[64, 32], help='Number of hidden units.')  # append to the last layer
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=300)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=16)
parser.add_argument("--datasets",type=str, default="A+Wacm")
parser.add_argument('--dim', type=int, default=16, help='Number of output dim.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--finetune_epoch', type=int, default=2000, help='Finetune Epoch.')
parser.add_argument('--finetune_lr', type=float, default=1e-3, help='Finetune learning rate.')
parser.add_argument('--weight', action='append', type=str, default=None, help='trained model weight')
parser.add_argument("--ratio", type=float, default=1.0)   # train dataset / all dataset
parser.add_argument("--root_dir", type=str, default='logs')   # output dir path
parser.add_argument("--time", type=bool, default=True)   # add time to store path
parser.add_argument("--save", type=bool, default=False)   # save the model or not
parser.add_argument("--plt_i", type=int, default=1000)   # plot interval
parser.add_argument("--few_shot", type=float, default=0.04)   # plot interval
parser.add_argument("--only", type=bool, default=False)   # only use the first dataset as target
parser.add_argument("--viz", type=bool, default=False)   # visualization
parser.add_argument("--gnn", type=str, default="gcn")   # choose to use which gnn: [gin, gcn, gat, graphsage]
parser.add_argument("--disc", type=str, default='1')   # domain discriminator type
parser.add_argument("--pre_finetune", type=int, default=0)   # fix diRedu and hidden, finetune classfier
parser.add_argument("--_lambda", action='append', type=str, default=None)   # weight for different signals
parser.add_argument("--_alpha", action='append', type=str, default=None)   # sudo weight for target domain
parser.add_argument("--alpha", type=float, default=1.0)   # in mixup beta distribution
parser.add_argument("--no_mixup", action='store_false', default=True)   # parameter in mixup
parser.add_argument("--no_balance", action='store_false', default=True)   # weight for cluster loss
parser.add_argument("--feat_num", type=int, default=100)   # number of reduced features
parser.add_argument("--assis_num", type=int, default=45)   # number of surrogate nodes


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

logger = get_logger(args.name)
log = Logger('TransNet', str(args.epoch) + '-' + str(args.finetune_epoch) + '-' + str(args.few_shot) +
             '-' + args.datasets + '-' + str(args.mu), root_dir=args.root_dir, with_timestamp=args.time)

log.add_params(vars(args))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def train_epoch(model, dataset, dataloader, optimizer, num_data_sets, args, source_insts, 
                target_insts, source_adj, source_labels, target_labels, target_adj, i, nbatch, device, 
                rate, sudo_weight, source_agent, target_agent, nclass, epoch):
    running_loss = 0.0
    dt_loss = 0.0
    domain_loss = 0.0
    few_shot_idx = dataset[i].finetune_indices
    dloaders = []
    model.train()
    for j in range(num_data_sets):
        dataset[j].change_task('train')
        dloaders.append(iter(dataloader[j]))

    for j in range(num_data_sets):
        if j != i:
            s_labels = torch.zeros(dataset[j].num, requires_grad=False).type(
                torch.LongTensor).to(device)
    t_labels = torch.ones(dataset[i].num, requires_grad=False).type(
        torch.LongTensor).to(device)


    for idx in range(min(nbatch)):
        sindex = []
        pos_u_idx = []
        pos_v_idx = []
        neg_v_idx = []
        for j in range(num_data_sets):
            if j != i:
                batch = next(dloaders[j])
                sindex.append(batch.to(device))
        batch = next(dloaders[i])
        tindex = batch.to(device)

        optimizer.zero_grad()
        svec, tvec, sh_linear, th_linear, sdomains, tdomains, sdomains_0, tdomains_0 = model(source_insts, target_insts, source_adj, target_adj, rate)

        if sudo_weight > 0:
            if epoch > 1000:
                relax=True
            else:
                relax=False
            target_agent.get_dicts(tvec, ass_idx, ass_label, relax)
            dt_losses = torch.stack([(source_agent[j].make_loss(svec[j]) + sudo_weight * target_agent.make_loss(tvec, target_agent.dicts)) for j in range(num_domains)])
        else:
            dt_losses = torch.stack([(source_agent[j].make_loss(svec[j])) for j in range(num_domains)])


        # Domain loss                                    
        if args.no_balance == True:
            selected_idxs = []
            selected_idx = []
            selected_label = []
            temp_idx = np.arange(source_labels[0].shape[0])
            for l_num in range(nclass[0]):
                s_idx = (source_labels[0] == l_num).cpu().numpy()
                if s_idx.sum() == 0:
                    continue
                s_idx = np.random.choice(temp_idx[s_idx], mini_count, replace=False)
                selected_idx += s_idx.tolist()
                selected_label += [l_num]
            selected_idxs.append(selected_idx)

            domain_losses = torch.stack([F.nll_loss(sdomains[j][selected_idxs[j]], s_labels[selected_idxs[j]]) +
                                        F.nll_loss(tdomains[j], t_labels)
                                        for j in range(num_domains)])
            domain_losses_0 = torch.stack([F.nll_loss(sdomains_0[j][selected_idxs[j]], s_labels[selected_idxs[j]]) +
                                        F.nll_loss(tdomains_0[j], t_labels)
                                        for j in range(num_domains)])
        else:
            domain_losses = torch.stack([F.nll_loss(sdomains[j], s_labels) +
                                        F.nll_loss(tdomains[j], t_labels)
                                        for j in range(num_domains)])
            domain_losses_0 = torch.stack([F.nll_loss(sdomains_0[j], s_labels) +
                                        F.nll_loss(tdomains_0[j], t_labels)
                                        for j in range(num_domains)])
        if args.disc == '2':
            domain_losses = domain_losses_0
        if args.disc == '3':
            domain_losses = domain_losses + 5*domain_losses_0

        loss = torch.max(dt_losses) + args.mu * torch.min(domain_losses)

        running_loss += loss.item()
        dt_loss += torch.max(dt_losses).item()
        domain_loss += torch.max(domain_losses).item()
        loss.backward()

        optimizer.step()

    return running_loss / min(nbatch), dt_loss / min(nbatch), domain_loss / min(nbatch), svec, tvec, sh_linear, th_linear


def vali_epoch(model, dataset, dataloader, optimizer, num_data_sets, args, source_insts, 
                target_insts, source_adj, source_labels, target_labels, target_adj, i, nbatch, device, 
                rate, sudo_weight, source_agent, target_agent, nclass, epoch):
    model.eval()
    nbatch = 0
    vali_loss = 0

    dloaders = []
    for j in range(num_data_sets):
        dataset[j].change_task('vali')
        dataloader = DataLoader(dataset[j], batch_size=len(dataset[j]),
                                shuffle=True, num_workers=0, collate_fn=dataset[j].collate, drop_last=True)
        nbatch = len(dataloader)
        dloaders.append(iter(dataloader))

    for j in range(num_data_sets):
        if j != i:
            s_labels = torch.zeros(dataset[j].num, requires_grad=False).type(
                torch.LongTensor).to(device)
    t_labels = torch.ones(dataset[i].num, requires_grad=False).type(
        torch.LongTensor).to(device)

    with torch.no_grad():
        svec, tvec, sh_linear, th_linear, sdomains, tdomains, sdomains_0, tdomains_0 = model(source_insts, target_insts, source_adj, target_adj, rate)

    for idx in range(nbatch):
        sindex = []
        pos_u_idx = []
        pos_v_idx = []
        neg_v_idx = []
        for j in range(num_data_sets):
            if j != i:
                batch = next(dloaders[j])
                sindex.append(batch.to(device))
        batch = next(dloaders[i])
        tindex = batch.to(device)

        if sudo_weight > 0:
            dt_losses = torch.stack([(source_agent[j].make_loss(svec[j], task='vali') + sudo_weight * target_agent.make_loss(tvec, target_agent.dicts, task='vali')) for j in range(num_domains)])
        else:
            dt_losses = torch.stack([(source_agent[j].make_loss(svec[j])) for j in range(num_domains)])

        domain_losses = torch.stack([F.nll_loss(sdomains[j], s_labels) +
                                    F.nll_loss(tdomains[j], t_labels)
                                    for j in range(num_domains)])
        domain_losses_0 = torch.stack([F.nll_loss(sdomains_0[j], s_labels) +
                                    F.nll_loss(tdomains_0[j], t_labels)
                                    for j in range(num_domains)])
        if args.disc == '2':
            domain_losses = domain_losses_0
        if args.disc == '3':
            domain_losses = domain_losses + 5*domain_losses_0

        loss = torch.max(dt_losses) + args.mu * torch.min(domain_losses)

        vali_loss += loss.item()

    return vali_loss/nbatch


def test(model, dataset, args, insts, labels, adj, device, index=-1):
    model.eval()
    _insts = insts.clone().detach().to(device)
    _labels = labels.clone().detach().cpu()
    with torch.no_grad():
        logprobs, vec, linear = model.inference(_insts, adj, index)
    preds_labels = torch.max(logprobs[dataset.test_indices], 1)[1].squeeze_().cpu()
    pred_acc = torch.sum(preds_labels == _labels[dataset.test_indices]).item() / float(len(dataset.test_indices))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mic,mac=f1_score(_labels[dataset.test_indices], preds_labels, average="micro"), f1_score(_labels[dataset.test_indices], preds_labels, average="macro")
        matrix = confusion_matrix(_labels[dataset.test_indices], preds_labels)
        recall_per_class = matrix.diagonal()/matrix.sum(axis=1)
        precision_per_class = matrix.diagonal()/matrix.sum(axis=0)

    return pred_acc, mic, mac, recall_per_class, precision_per_class


def test_source(model, dataset, args, insts, labels, adj, device, index=-1):
    model.eval()
    _insts = insts.clone().detach().to(device)
    _labels = labels.clone().detach().cpu()
    with torch.no_grad():
        logprobs, vec, linear = model.inference(_insts, adj, index)
    preds_labels = torch.max(logprobs[dataset.s_test_indices], 1)[1].squeeze_().cpu()
    pred_acc = torch.sum(preds_labels == _labels[dataset.s_test_indices]).item() / float(len(dataset.s_test_indices))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mic,mac=f1_score(_labels[dataset.s_test_indices], preds_labels, average="micro"), f1_score(_labels[dataset.s_test_indices], preds_labels, average="macro")
        matrix = confusion_matrix(_labels[dataset.s_test_indices], preds_labels)
        recall_per_class = matrix.diagonal()/matrix.sum(axis=1)
        precision_per_class = matrix.diagonal()/matrix.sum(axis=0)

    return pred_acc, mic, mac, recall_per_class, precision_per_class

def fine_tune(model, dataset, args, target_insts, target_labels, target_adj, i, device, graph, few_shot_labels):
    dataset.change_task('finetune')
    model.finetune()
    finetune_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.finetune_lr, weight_decay=args.weight_decay)
    t_insts = target_insts.clone().detach().to(device)
    if args.batch_size == -1:
        bt_size = len(dataset)
    else:
        bt_size = args.batch_size
    f_data_loader = DataLoader(dataset, batch_size=bt_size,
                                shuffle=False, num_workers=0, collate_fn=dataset.collate)

    pred_acc, mic, mac, recall_per_class, precision_per_class = test(model, dataset, args, target_insts, target_labels, target_adj, device, -1)
    log.add_metric(graph.name + '_pred_acc', pred_acc, 0)
    log.add_metric(graph.name + '_micro_F', mic, 0)
    log.add_metric(graph.name + '_macro_F', mac, 0)
    log.add_metric(graph.name + '_recall_per_class', recall_per_class, 0)
    log.add_metric(graph.name + '_precision_per_class', precision_per_class, 0)

    best_target_acc = 0.0
    best_epoch = 0.0
    for epoch in range(args.finetune_epoch):
        model.train()
        if epoch == args.pre_finetune:
            model.finetune_inv()
            finetune_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.finetune_lr, weight_decay=args.weight_decay)
        running_loss = 0
        dataset.change_task('finetune')
        for batch in f_data_loader:
            finetune_optimizer.zero_grad()
            tindex = batch.to(device)
            logprobs, tvec, tlinear = model.inference(t_insts, target_adj, index=-1)
            loss = F.nll_loss(logprobs[tindex], target_labels[tindex])
            running_loss += loss.item()
            loss.backward()
            finetune_optimizer.step()

        pred_acc, mic, mac, recall_per_class, precision_per_class = test(model, dataset, args, target_insts, target_labels, target_adj, device, -1)
        logger.info("Iteration {}, loss = {}, acc = {}".format(epoch, running_loss/len(f_data_loader), pred_acc))
        log.add_metric(graph.name+'_finetuine_loss', running_loss/len(f_data_loader), epoch)
        log.add_metric(graph.name + '_pred_acc', pred_acc, epoch+1)
        log.add_metric(graph.name + '_micro_F', mic, epoch+1)
        log.add_metric(graph.name + '_macro_F', mac, epoch+1)
        log.add_metric(graph.name + '_recall_per_class', recall_per_class, epoch+1)
        log.add_metric(graph.name + '_precision_per_class', precision_per_class, epoch+1)
        if pred_acc > best_target_acc:
            best_target_acc = pred_acc
            best_epoch = epoch
            viz_tvec = tvec

    if args.viz == True:
        viz_single(viz_tvec, log._logdir, graph.name, best_epoch-1, best_target_acc, target_labels, few_shot_labels)
    print("=============================================================")
    line = "{} - Epoch: {}, best_target_acc: {}"\
        .format(graph.name, best_epoch, best_target_acc)
    print(line)

    return best_target_acc


def agnn_fine_tune(model, dataset, args, target_insts, target_labels, target_adj, i, device, graph, few_shot_labels):
    dataset.change_task('finetune')
    finetune_optimizer = optim.Adam(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
    t_insts = target_insts.clone().detach().to(device)
    if args.batch_size == -1:
        bt_size = len(dataset)
    else:
        bt_size = args.batch_size
    f_data_loader = DataLoader(dataset, batch_size=bt_size,
                                shuffle=True, num_workers=0, collate_fn=dataset.collate)

    pred_acc, mic, mac, recall_per_class, precision_per_class = test(model, dataset, args, target_insts, target_labels, target_adj, device, -1)
    best_target_acc = 0.0
    best_epoch = 0.0
    for epoch in tqdm(range(1000)):
        model.train()
        running_loss = 0
        dataset.change_task('finetune')
        for batch in f_data_loader:
            finetune_optimizer.zero_grad()
            tindex = batch.to(device)
            logprobs, tvec = model(t_insts, target_adj, index=-1)
            loss = F.nll_loss(logprobs[tindex], target_labels[tindex])
            running_loss += loss.item()
            loss.backward()
            finetune_optimizer.step()

        pred_acc, mic, mac, recall_per_class, precision_per_class = test(model, dataset, args, target_insts, target_labels, target_adj, device, -1)
    print("=============================================================")
    line = "{} - Epoch: {}, acc: {}"\
        .format(graph.name, epoch, pred_acc)
    print(line)

    return pred_acc


# Loading the randomly partition the amazon data set.
time_start = time.time()

globel_number = 1
graphs = []
num_nodes, input_dim = [], []
_alpha, _lambda = [], []
datasets = args.datasets.split("+")
for i, data in enumerate(datasets):
    if args._alpha is None:
        _alpha.append(0)
    else:
        _alpha.append(float(args._alpha[i]))
    if args._lambda is None:
        _lambda.append(0)
    else:
        _lambda.append(float(args._lambda[i]))
    g = GraphLoader(data, sparse=True, args=args)
    g.process()
    graphs.append(g)
    num_nodes.append(g.X.shape[0])
    input_dim.append(g.X.shape[1])

dataset_ = []
data_name = []
for g in graphs:
    actual_adj, hide_adj, conn, conn_hide, conn_actual, frequency = process_data(g, g.name)
    # actual_adj = g.adj.to_dense().cpu().numpy()
    data = MyData(actual_adj, g.Y, args.seed)
    # print(data.num)
    dataset_.append(MyDataset(data, "train", ratio=args.ratio, few_shot=args.few_shot, seed=args.seed))
    data_name.append(g.name)

n_nodes = min(num_nodes)
time_end = time.time()
logger.info("Time used to process the data set = {} seconds.".format(time_end - time_start))

num_data_sets = len(dataset_)

for i in range(num_data_sets):
    dataset = []
    for j in range(len(dataset_)):
        dataset.append(copy.deepcopy(dataset_[j]))
    nclass, dataloader, nbatch, ndim = [], [], [], []
    # get statistics of every domain
    for j in range(num_data_sets):
        if j != i:
            nclass.append(graphs[j].Y.cpu().numpy().max()+1)
            ndim.append(int(graphs[j].X.shape[1]))
        dataset[j].change_task('train')
        dataloader.append(DataLoader(dataset[j], batch_size=len(dataset[j]),
                                        shuffle=True, num_workers=0, collate_fn=dataset[j].collate, drop_last=True))
        nbatch.append(len(dataloader[-1]))
    nclass.append(graphs[i].Y.cpu().numpy().max()+1)
    ndim.append(int(graphs[i].X.shape[1]))
    
    selected_idx = []
    selected_label = []
    if nclass[-1] < nclass[0]:
        for j in range(num_data_sets):
            if j != i:
                uniques, counts = graphs[j].Y.unique(return_counts=True)
                uniques = uniques.cpu().numpy()
                temp_idx = np.arange(graphs[j].Y.shape[0])
                for l_num in range(nclass[-1]):
                    s_label = uniques[counts.argmax()]
                    s_idx = (graphs[j].Y == s_label).cpu().numpy()
                    selected_idx += temp_idx[s_idx].tolist()
                    selected_label += [s_label]
                    counts[counts.argmax()] = 0

                new_label = 0
                np.random.shuffle(selected_idx)
                nclass[0] = nclass[-1]
                graphs[j].X = graphs[j].X[selected_idx]
                graphs[j].normadj = graphs[j].normadj.to_dense()[selected_idx,:][:,selected_idx].to_sparse()
                graphs[j].adj = graphs[j].adj.to_dense()[selected_idx,:][:,selected_idx].to_sparse()
                graphs[j].G = nx.from_numpy_matrix(graphs[j].adj.to_dense().cpu().numpy())
                graphs[j].Y = graphs[j].Y[selected_idx]
                copy_labels = copy.deepcopy(graphs[j].Y)
                for s_label in selected_label:
                    graphs[j].Y[copy_labels == s_label] = new_label
                    new_label += 1
                data = MyData(graphs[j].normadj.to_dense(), graphs[j].Y, args.seed)
                dataset[j] = MyDataset(data, "train", ratio=args.ratio, few_shot=args.few_shot, seed=args.seed)
                dataloader[j] = DataLoader(dataset[j], batch_size=len(dataset[j]),
                                        shuffle=True, num_workers=0, collate_fn=dataset[j].collate, drop_last=True)

    # Build source instances.
    source_insts = []
    source_adj = []
    source_labels = []
    source_names = []
    source_agent = []

    for j in range(num_data_sets):
        if j != i:
            source_insts.append(graphs[j].X)
            source_adj.append(graphs[j].normadj.to_dense())
            source_labels.append(graphs[j].Y)
            source_names.append(graphs[j].name)

    # Build target instances.
    target_idx = i
    target_insts = graphs[i].X
    target_adj = graphs[i].normadj.to_dense()
    target_labels = graphs[i].Y
        
    mini_count = 999999999
    for j in range(num_data_sets):
        if j != i:
            uniques, counts = source_labels[0].unique(return_counts=True)
            if counts.cpu().numpy().min() < mini_count:
                mini_count = counts.cpu().numpy().min()

    # set parameters for mdanet
    configs = {"input_dim": ndim, "hidden_layers": args.hidden, "num_classes": nclass,
                "num_epochs": args.epoch, "batch_size": args.batch_size, "lr": args.lr, "mu": args.mu, "num_domains":
                    num_data_sets - 1, "ndim": args.dim,
                "dropout": args.dropout, "finetune epoch": args.finetune_epoch, "train ratio": args.ratio,
                "num_nodes": n_nodes, "type": args.gnn, "feat_num": args.feat_num}
    num_epochs = configs["num_epochs"]
    batch_size = configs["batch_size"]
    num_domains = configs["num_domains"]
    lr = configs["lr"]
    mu = configs["mu"]
    logger.info("Target domain is {}.".format(graphs[i].name))
    logger.info("Hyperparameter setting = {}.".format(configs))

    # For visualization
    few_shot_labels = target_labels.clone()*0-1
    few_shot_labels[dataset[i].finetune_indices] = target_labels[dataset[i].finetune_indices]
    # Train DannNet.
    transnet = TransNet(configs).to(device)
    target_agent = GeneralSignal(graphs[i], graphs[i].adj, target_labels, target_insts, dataset=dataset[i], 
            nhid=args.dim, device=device, args = args, seed=args.seed, model=transnet, index=-1, _lambda=_lambda[i])
    jj = 0
    for j in range(num_data_sets):
        if j != i:
            source_agent.append(GeneralSignal(graphs[j], graphs[j].adj, source_labels[jj], source_insts[jj], dataset=dataset[j], 
                    nhid=args.dim, device=device, args = args, seed=args.seed, model=transnet, index=jj, _lambda=_lambda[i]))
            jj += 1
    optimizer = optim.Adam(transnet.parameters(), lr=lr, betas=(0.5, 0.999))

    if args.weight is None:
        # Train an assistant GNN
        print("########### Train assistant GNN ###################")
        agnn_configs = {"input_dim": ndim[-1], "hidden_layers": args.hidden, "num_classes": nclass[-1],
                "num_epochs": args.epoch, "batch_size": args.batch_size, "lr": args.lr, "mu": args.mu, "num_domains":
                    num_data_sets - 1, "ndim": args.dim,
                "dropout": args.dropout, "finetune epoch": args.finetune_epoch, "train ratio": args.ratio,
                "num_nodes": n_nodes, "type": args.gnn}
        agnn = BASE(agnn_configs).to(device)
        agnn_optimizer = optim.Adam(agnn.parameters(), lr=lr, betas=(0.5, 0.999))

        pred_acc = agnn_fine_tune(agnn, dataset[i], args, target_insts, target_labels, target_adj, i, device, graphs[i], few_shot_labels)
        print("########### Finish assistant GNN ###################")


        ############ Get assistant nodes  ###################
        with torch.no_grad():
            logprobs, _, _ = agnn.inference(target_insts, target_adj)
        preds_labels = torch.max(logprobs, 1)[1].squeeze_().detach().cpu()
        all_idx = torch.tensor([i for i in range(logprobs.shape[0])])
        logmax_i = logprobs.max(1)[1]
        logmax_v = logprobs.max(1)[0]
        ass_nodes = []
        ass_labels = []
        ass_size = []
        few_shot_idx = dataset[i].finetune_indices
        for idx in range(nclass[-1]):
            t_idx = all_idx[logmax_i == idx][torch.sort(logmax_v[logmax_i == idx], descending=True)[1]]
            ass_size.append(t_idx)
            if t_idx.shape[0] < args.assis_num:
                ass_nodes.append(torch.tensor([i.item() for i in t_idx if i not in few_shot_idx]))
            else:
                ass_nodes.append(torch.tensor([i.item() for i in t_idx[:args.assis_num] if i not in few_shot_idx]))
            ass_labels.append(torch.ones(ass_nodes[-1].shape, dtype=int)*idx)
        ass_idx = torch.cat([i for i in ass_nodes])
        ass_label =  torch.cat([i for i in ass_labels])
        ass_labels = target_labels.clone()*0-1
        ass_labels[ass_idx] = ass_label.to(device)
        ######################################################

        
        best_target_acc = 0.0
        best_epoch = 0.0
        time_start = time.time()
        for t in range(num_epochs):
            # rate = min((t + 1) / num_epochs, 0.05)
            p = float(1 + t) / num_epochs
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            sudo_weight = _alpha[i]
            
            train_loss, dt_loss, domain_loss, svec, tvec, sh_linear, th_linear = train_epoch(transnet, dataset, dataloader, optimizer, num_data_sets,
                                                                                            args, source_insts, target_insts, source_adj, source_labels, target_labels,
                                                                                            target_adj, i, nbatch, device, alpha, sudo_weight, source_agent, target_agent, nclass, t)
            vali_loss = vali_epoch(transnet, dataset, dataloader, optimizer, num_data_sets,
                                    args, source_insts, target_insts, source_adj, source_labels, target_labels,
                                    target_adj, i, nbatch, device, alpha, sudo_weight, source_agent, target_agent, nclass, t)

            # Test on source domains
            source_acc = {}
            jj = 0
            for j in range(num_data_sets):
                if j != i:
                    pred_acc, mic, mac, recall_per_class, precision_per_class = test_source(transnet, dataset[j], args, source_insts[jj], 
                        source_labels[jj], source_adj[jj], device, index=jj)
                    source_acc[source_names[jj]] = pred_acc
                    log.add_metric(graphs[i].name+'_'+source_names[jj]+'_prediction_accuracy', pred_acc, t)
                    jj = jj + 1
            log.add_metric(graphs[i].name + '_da_prediction_accuracy', pred_acc, t)
            log.add_metric(graphs[i].name + '_da_micro_F', mic, t)
            log.add_metric(graphs[i].name + '_da_macro_F', mac, t)
            log.add_metric(graphs[i].name + '_da_recall_per_class', recall_per_class, t)
            log.add_metric(graphs[i].name + '_da_precision_per_class', precision_per_class, t)

            if pred_acc > best_target_acc:
                best_target_acc = pred_acc
                best_epoch = t

            logger.info("Epoch {}, train loss = {}, validation loss = {}, source_acc = {}".format(
                t, train_loss, vali_loss, source_acc))

            log.add_metric(graphs[i].name+'_dt_loss', dt_loss, t)
            log.add_metric(graphs[i].name+'_domain_loss', domain_loss, t)
            log.add_metric(graphs[i].name+'_train_loss', train_loss, t)
            log.add_metric(graphs[i].name+'_vali_loss', vali_loss, t)

            if t>0 and t % args.plt_i == 0 and args.viz == True:
                with torch.no_grad():
                    logprobs, _, _ = transnet.inference(target_insts, target_adj, index=-1, sudo=True)
                preds_labels = torch.max(logprobs, 1)[1].squeeze_().detach().cpu()
                viz_single(tvec, log._logdir, graphs[i].name, t, 0.0, target_labels, few_shot_labels, adapt=1, sudo_label=preds_labels, ass_node=ass_labels)

            # Save weight
            if t == num_epochs - 1 and args.save is True:
                checkpoint = {"model_state_dict": transnet.state_dict(),
                                "optimizer_state_dic": optimizer.state_dict(),
                                "loss": train_loss,
                                "epoch": t}
                path_checkpoint = "checkpoint_{}".format(t)
                if not os.path.exists(os.path.join(log._logdir, graphs[i].name)):
                    os.makedirs(os.path.join(log._logdir, graphs[i].name))
                torch.save(checkpoint, os.path.join(log._logdir, graphs[i].name, path_checkpoint))
        if num_epochs > 0 and args.viz == True:
            with torch.no_grad():
                logprobs, _, _ = transnet.inference(target_insts, target_adj, index=-1, sudo=True)
            preds_labels = torch.max(logprobs, 1)[1].squeeze_().detach().cpu()
            viz_single(tvec, log._logdir, graphs[i].name, t+1, 0.0, target_labels, few_shot_labels, adapt=1, sudo_label=preds_labels, ass_node=ass_labels)
        print("=============================================================")
        line = "{} - Epoch: {}, best_target_acc: {}"\
            .format(graphs[i].name, best_epoch, best_target_acc)
        print(line)
        time_end = time.time()
    else:
        print('Recovering from %s ...' % (args.weight[i]))
        checkpoint = torch.load(args.weight[i])
        epoch = checkpoint["epoch"]

        transnet.load_state_dict(checkpoint["model_state_dict"])

        few_shot_labels = target_labels.clone()*0-1
        few_shot_labels[dataset[i].finetune_indices] = target_labels[dataset[i].finetune_indices]

    # Finetune on target domain
    pred_acc = fine_tune(transnet, dataset[i], args, target_insts, target_labels, target_adj, i, device, graphs[i], few_shot_labels)

    logger.info("label prediction accuracy on {} = {}, time used = {} seconds.".
                format(data_name[i], pred_acc, time_end - time_start))

    dataset[i].change_task('train')
    if args.only == True:
        break
logger.info("*" * 100)



