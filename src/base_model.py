import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from gnns import GNN
import numpy as np
import random
from random import sample
import networkx as nx
import collections
import pdb


logger = logging.getLogger(__name__)


class BASE(nn.Module):
    """
    Multi-layer perceptron with adversarial regularizer by domain classification.
    """
    def __init__(self, configs):
        super(BASE, self).__init__()
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.hiddens = GNN(nfeat=configs["input_dim"],
                              nhid=configs["hidden_layers"],
                              nclass=configs["num_classes"],
                              ndim=configs["ndim"],
                              gnn_type=configs["type"],
                              bias=None,
                              dropout=configs["dropout"]
                              )
        self.softmax = nn.Linear(configs["ndim"], configs["num_classes"])

    def forward(self, tinput, adj, index=-1):
        h_relu = tinput.clone()
        h_relu = self.hiddens(h_relu, adj)
        logprobs = F.log_softmax(h_relu, dim=1)
        return logprobs, h_relu

    def inference(self, tinput, adj, index=-1):
        h_relu = tinput.clone()
        h_relu = self.hiddens(h_relu, adj)
        logprobs = F.log_softmax(h_relu, dim=1)
        return logprobs, h_relu, h_relu