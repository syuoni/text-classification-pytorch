# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import reinit_parameters_


def construct_pooling_layer(pooling_kwargs):
    if pooling_kwargs['type'] == 'max':
        layer = MaxPoolingLayer()
    elif pooling_kwargs['type'] == 'mean':
        layer = MeanPoolingLayer()
    elif pooling_kwargs['type'] == 'attention':
        layer = AttenLayer(pooling_kwargs['hidden_dim'], pooling_kwargs['atten_dim'])
    else:
        raise Exception('Invalid Pooling type!', pooling_kwargs['type'])
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
        proj = F.tanh(self.project(nn_outs))
        
        # atten: tensor (float32) of shape (batch, step)
        # torch.matmul: batched matrix x broadcasted vector
        atten = proj.matmul(self.context)
        # atten_outs: tensor (float32) of (batch, hidden)
        atten_outs = []
        for batch_idx in range(atten.size(0)):
            this_atten = F.softmax(atten[batch_idx, :batch_lens[batch_idx].item()], dim=-1)
            this_nn_outs = nn_outs[batch_idx, :batch_lens[batch_idx].item()]
            atten_outs.append(this_nn_outs.transpose(1, 0).mv(this_atten))
        return torch.stack(atten_outs)
    

class HiePoolingLayer(nn.Module):
    def __init__(self, pooling_layer):
        super(HiePoolingLayer, self).__init__()
        self.pooling_layer = pooling_layer
            
    def forward(self, nn_outs, batch_x):
        '''
        Input:
            nn_outs: tensor (float32) of shape (batch, step, emb)
            batch_x: tensor (int64) of shape (batch, step)
        Return:
            interm_outs: tensor (float32) of shape (batch, interm_step, hidden)
            interm_batch_lens: tensor (int64) of shape (batch, )
        '''
        interm_hidden_list = []
        interm_batch_lens = []        
        for batch_idx in range(batch_x.size(0)):            
            cuts = torch.arange(batch_x.size(1), dtype=torch.int64, device=batch_x.device)[batch_x[batch_idx]==1] + 1
            
            sub_pooled_outs_list = []
            for cut_idx in range(cuts.size(0)):
                cut0 = 0 if cut_idx == 0 else cuts[cut_idx-1]
#                print(batch_x[batch_idx, cut0:cuts[cut_idx]][-1])
                
                # sub_outs: tensor (float32) of shape (sub_step, hidden)
                sub_outs = nn_outs[batch_idx, cut0:cuts[cut_idx]]
                # sub_pooled_outs: tensor (float32) of shape (1, hidden)
                sub_pooled_outs = self.pooling_layer(sub_outs.unsqueeze(0), 
                                                     torch.tensor([sub_outs.size(0)], device=batch_x.device))
                sub_pooled_outs_list.append(sub_pooled_outs)
                
#            import pdb; pdb.set_trace()
            interm_hidden = torch.cat(sub_pooled_outs_list)
            interm_hidden_list.append(interm_hidden)
            interm_batch_lens.append(interm_hidden.size(0))
            
        maxlen = max(interm_batch_lens)
        interm_hidden_list = [F.pad(interm_hidden_list[i], (0, 0, 0, maxlen-interm_batch_lens[i])) for i in range(batch_x.size(0))]
        
#        for x in interm_hidden_list:
#            print(x.size())
        
        return torch.stack(interm_hidden_list), torch.tensor(interm_batch_lens, device=batch_x.device)
    
    