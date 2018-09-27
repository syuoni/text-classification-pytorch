# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import reinit_parameters_


def construct_pooling_layer(pooling_type, pooling_kwargs):
    if pooling_type == 'max':
        layer = MaxPoolingLayer()
    elif pooling_type == 'mean':
        layer = MeanPoolingLayer()
    elif pooling_type == 'attention':
        layer = AttenLayer(**pooling_kwargs)
    else:
        raise Exception('Invalid Pooling type!', pooling_type)
    return layer


class MeanPoolingLayer(nn.Module):
    def __init__(self):
        super(MeanPoolingLayer, self).__init__()
        
    def forward(self, nn_outs, batch_lens):
        '''
        Input:
            nn_outs: tensor (float32) of shape (batch, step, hidden)
            batch_lens: tensor (int64) of shape (batch, )
        Return:
            pooled_outs: tensor (float32) of (batch, hidden)
        '''
        pooled_outs = nn_outs.sum(dim=1) / batch_lens.view(-1, 1).type(torch.float32)
        return pooled_outs
    
    
class MaxPoolingLayer(nn.Module):
    def __init__(self):
        super(MaxPoolingLayer, self).__init__()
        
    def forward(self, nn_outs, batch_lens):
        '''
        Input:
            nn_outs: tensor (float32) of shape (batch, step, hidden)
            batch_lens: tensor (int64) of shape (batch, )
        Return:
            pooled_outs: tensor (float32) of (batch, hidden)
        '''
        pooled_outs, _ = nn_outs.max(dim=1)
        return pooled_outs


class AttenLayer(nn.Module):
    def __init__(self, hidden_dim, atten_dim=None):
        super(AttenLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.atten_dim = hidden_dim if atten_dim is None else atten_dim
        
        self.context = nn.Parameter(torch.empty(self.atten_dim).uniform_(-0.1, 0.1), requires_grad=True)
        self.project = nn.Linear(hidden_dim, self.atten_dim)
        reinit_parameters_(self.project)
        
    def forward(self, nn_outs, batch_lens):
        '''
        Input:
            nn_outs: tensor (float32) of shape (batch, step, hidden)
            batch_lens: tensor (int64) of shape (batch, )
        Return:
            atten_outs: tensor (float32) of (batch, hidden)
        '''
        # proj: tensor (float32) of shape (batch, step, atten)
        proj = torch.tanh(self.project(nn_outs))
        
        # atten: tensor (float32) of shape (batch, step)
        # torch.matmul: batched matrix x broadcasted vector
        atten = proj.matmul(self.context)
        
#        atten_outs = []
#        for batch_idx in range(atten.size(0)):
#            this_atten = F.softmax(atten[batch_idx, :batch_lens[batch_idx].item()], dim=-1)
#            this_nn_outs = nn_outs[batch_idx, :batch_lens[batch_idx].item()]
#            atten_outs.append(this_nn_outs.transpose(1, 0).mv(this_atten))
#        atten_outs = torch.stack(atten_outs)
        
        # NOTE: use requires_grad=False to avoid it being a "leaf variable". 
        # atten_softmax: tensor (float32) of shape (batch, step)
        atten_softmax = torch.zeros_like(atten, requires_grad=False)
        for batch_idx in range(atten.size(0)):
            atten_softmax[batch_idx, :batch_lens[batch_idx].item()] = F.softmax(atten[batch_idx, :batch_lens[batch_idx].item()], dim=-1)
        
        # atten_outs: tensor (float32) of (batch, hidden)
        atten_outs = atten_softmax.unsqueeze(1).matmul(nn_outs).squeeze(1)
        return atten_outs
    
    