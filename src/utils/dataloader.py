import os
import networkx as nx
import pickle as pkl
import torch
import numpy as np
from collections import OrderedDict
import time
from utils.utils import *
from torch.utils.data import Dataset
from sklearn import preprocessing
import json
import scipy.sparse
from torch_geometric.io import read_txt_array
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Amazon
import os.path as osp

import pdb

# Set random number seed.


class GraphLoader(object):

    def __init__(self,name,root = "./data",undirected=True, hasX=True,hasY=True,header=True,sparse=True,multigraphindex=None,args=None):

        if name == "reddit1401":
            self.name = name + '_' + multigraphindex
        else:
            self.name = name
        self.undirected = undirected
        self.hasX = hasX
        self.hasY = hasY
        self.header = header
        self.sparse = sparse
        if len(name) > 2 and name[0] != '_' and name[0] != 'W':
            self.dirname = os.path.join(root,name)
        else:
            self.dirname = os.path.join(root,'input')
        if name == "reddit1401":
            self.prefix = os.path.join(root, name, multigraphindex, multigraphindex)
        elif name[0] == '_':
            self.prefix = os.path.join(root, name[1:])
        elif name == 'C':
            self.prefix = os.path.join(root,'input/'+'citationv1'+'.mat')
        elif name == 'A':
            self.prefix = os.path.join(root,'input/'+'acmv9'+'.mat')
        elif name == 'D':
            self.prefix = os.path.join(root,'input/'+'dblpv7'+'.mat')
        elif name == 'B1':
            self.prefix = os.path.join(root,'input/'+'Blog1'+'.mat')   
        elif name == 'B2':
            self.prefix = os.path.join(root,'input/'+'Blog2'+'.mat')   
        else:
            self.prefix = os.path.join(root, name, name)
        self._load()
        self._registerStat()


    def _loadConfig(self):
        file_name = os.path.join(self.dirname,"bestconfig.txt")
        f = open(file_name,'r')
        L = f.readlines()
        L = [x.strip().split() for x in L]
        self.bestconfig = {x[0]:x[1] for x in L if len(x)!=0}


    def _loadGraph(self, header = True):
        """
            load file in form:
            --------------------
            NUM_Of_NODE\n
            v1 v2\n
            v3 v4\n
            --------------------
        """
        file_name = self.prefix+".edgelist"
        if not header:
            logger.warning("You are reading an edgelist with no explicit number of nodes")
        if self.undirected:
            G = nx.Graph()
        else:
            G = nx.DiGraph()
        with open(file_name) as f:
            L = f.readlines()
            if header:
                num_node = int(L[0].strip())
                L = L[1:]
            edge_list = [[int(x) for x in e.strip().split()] for e in L]
            nodeset = set([x for e in edge_list for x in e])
        
        if header:
            G.add_nodes_from([x for x in range(num_node)])
        else:
            G.add_nodes_from([x for x in range(max(nodeset)+1)])
        G.add_edges_from(edge_list)
        self.G = G
        self.edges = torch.tensor(np.array(G.edges).T, dtype=torch.long).cuda()

    def _loadX(self):
        self.X = pkl.load(open(self.prefix + ".x.pkl", 'rb'))
        self.X = self.X.astype(np.float32)
        if self.name in  ["coauthor_phy","corafull"]:
            self.X = self.X[:,:2000] # the coauthor_phy's feature is too large to fit in the memory.

    def _loadY(self):
        self.Y = pkl.load(open(self.prefix+".y.pkl",'rb'))#.astype(np.float32)

    def _getAdj(self):
        self.adj = nx.adjacency_matrix(self.G).astype(np.float32)

    def _toTensor(self,device=None):
        if device is None:
            if self.sparse:
                self.adj = sparse_mx_to_torch_sparse_tensor(self.adj).cuda()
                self.normadj = sparse_mx_to_torch_sparse_tensor(self.normadj).cuda()
            else:
                self.adj = torch.from_numpy(self.adj).cuda()
                self.normadj = torch.from_numpy(self.normadj).cuda()
            self.X = torch.from_numpy(self.X).cuda()
            self.Y = torch.from_numpy(self.Y).cuda()

    def _load(self):
        if self.name[0] == '_':
            if self.name == '_com':
                dataset = Amazon(root='data/', name='computers')
            elif self.name == '_pt':
                dataset = Amazon(root='data/', name='photo')
            graph = dataset[0]
            self.X = graph.x.numpy().astype(np.float32)
            self.Y = graph.y.numpy()
            np.save(self.name+'_x', self.X)
            np.save(self.name+'_y', self.Y)
            self.G = nx.Graph()
            for i in range(self.Y.shape[0]):
                self.G.add_node(i)
            self.G.add_edges_from(graph.edge_index.numpy().T)

            self._loadConfig()
            self.edges = torch.tensor(np.array(self.G.edges).T, dtype=torch.long).cuda()  
            self._getAdj()

        elif self.name[0] == 'W':
            raw_dir = 'data/WWW'
            edge_path = osp.join(raw_dir, '{}_edgelist.txt'.format(self.name[1:]))
            edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long)
            self.X = np.load(osp.join(raw_dir, self.name+'_x.npy'))
            self.Y = np.load(osp.join(raw_dir, self.name+'_y.npy'))
            self.G = nx.Graph()
            for i in range(self.Y.shape[0]):
                self.G.add_node(i)
            self.G.add_edges_from(edge_index.numpy())
            self._loadConfig()
            self.edges = torch.tensor(np.array(self.G.edges).T, dtype=torch.long).cuda()  
            self._getAdj()
        
        elif len(self.name) > 2:
            self._loadGraph(header=self.header)
            self._loadConfig()
            if self.hasX:
                self._loadX()
            if self.hasY:
                self._loadY()
            self._getAdj()

        else:
            A, X, Y = load_network(self.prefix)
            self.X = np.array(X.astype(np.float32))
            self.X = self.X[:, self.X.sum(0) != 0]
            self.Y = Y
            self.G = nx.from_numpy_matrix(A)
            self._loadConfig()
            self.edges = torch.tensor(np.array(self.G.edges).T, dtype=torch.long).cuda()  
            self._getAdj()
        

    def _registerStat(self):
        L=OrderedDict()
        L["name"] = self.name
        L["nnode"] = self.G.number_of_nodes()
        L["nedge"] = self.G.number_of_edges()
        L["nfeat"] = self.X.shape[1]
        L["nclass"] = self.Y.max() + 1
        L["sparse"] = self.sparse
        L["multilabel"] = False
        L.update(self.bestconfig)
        self.stat = L

    def process(self):
        if int(self.bestconfig['feature_normalize']):
            self.X = column_normalize(preprocess_features(self.X))
        self.normadj = preprocess_adj(self.adj)
        if not self.sparse:
            self.adj = self.adj.todense()
            self.normadj = self.normadj.todense()
        self._toTensor()
        
        self.normdeg = self._getNormDeg()

    def _getNormDeg(self):
        self.deg = torch.sparse.sum(self.adj, dim=1).to_dense()
        normdeg =self.deg/ self.deg.max()
        return normdeg


# -----------------------------------------------------------------------------------------------------------------
class MyData:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, actual_adj, labels, seed=123):
        self.actual_adj = actual_adj
        self.seed = seed
        self.num = self.actual_adj.shape[0]
        self.labels = labels.cpu().numpy()


# -----------------------------------------------------------------------------------------------------------------
class MyDataset(Dataset):
    def __init__(self, data, task, ratio=2.0/3.0, few_shot=5, seed=123):
        self.data = data
        self.task = task
        self.seed = seed
        self.ratio = ratio
        self.few_shot = few_shot
        self.rand_index = []
        self.num = self.data.num
        self.indices = list(range(int(self.num)))
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        self.s_test_indices = []
        self.finetune_indices = []
        self.cross_validation(0)
        self.select_fintune()

    def __len__(self):
        if self.task == "train":
            return int(self.num * self.ratio * 0.9)
        elif self.task == "vali":
            return int(self.num * self.ratio) - int(self.num * self.ratio * 0.9)
        elif self.task == "test":
            if self.few_shot >= 1:
                return self.num - int(self.few_shot*len(set(self.data.labels)))
            else:
                return int(self.num * (1 - self.few_shot))
        elif self.task == "finetune":
            if self.few_shot >= 1:
                return int(self.few_shot*len(set(self.data.labels)))
            else:
                return int(self.num * self.few_shot)
        elif self.task == "s_test":
            if self.ratio == 1:
                return int(self.num * self.ratio) - int(self.num * self.ratio * 0.9)
            else:
                return int(self.num * (1-self.ratio))

    def __getitem__(self, idx):
        if self.task == "train":
            return self.train_indices[idx]
        elif self.task == "vali":
            return self.val_indices[idx]
        elif self.task == "test":
            return self.test_indices[idx]
        elif self.task == "finetune":
            return self.finetune_indices[idx]
        elif self.task == "s_test":
            return self.s_test_indices[idx]

    def change_task(self, task):
        self.task = task
        
    def cross_validation(self, fold):
        np.random.seed(self.seed)
        np.random.shuffle(self.indices)
        self.train_indices = self.indices[:int(self.num * self.ratio * 0.9)]
        self.val_indices = self.indices[int(self.num * self.ratio * 0.9):int(self.num * self.ratio)]

    def select_fintune(self):
        np.random.seed(self.seed)
        if self.few_shot >= 1:
            group = {}
            for i in set(self.data.labels):
                group[i] = []
                for j in range(len(self.data.labels)):
                    samp = np.random.choice(self.train_indices, 1)[0]
                    if self.data.labels[samp] == i:
                        group[i].extend([samp])
                    if len(group[i]) == self.few_shot:
                        break
            for i in set(self.data.labels):
                self.finetune_indices += group[i]
        else:
            self.finetune_indices = np.random.choice(self.train_indices, int(self.num*self.few_shot))
        self.test_indices = [x for x in list(range(self.num)) if x not in self.finetune_indices]
        if self.ratio == 1:
            self.s_test_indices = self.val_indices
        else:
            self.s_test_indices = [x for x in list(range(self.num)) if ((x not in self.train_indices) and (x not in self.val_indices))]
        np.random.shuffle(self.finetune_indices)
        

    def collate(self, batches):
        idx = [batch for batch in batches]
        idx = torch.LongTensor(idx)
        return idx


