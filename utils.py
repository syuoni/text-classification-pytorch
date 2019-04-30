# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def cast_to_tensor(batch, device):
    '''
    Input:
        batch: (batch_x, batch_y, batch_lens)
        batch_x: list of list of int
        batch_y: list of int
        batch_lens: list of int
        
    Return: tensors (int64)
    '''
    batch_x, batch_y, batch_lens = batch
    # Pad batch_x to have the same lengths
    maxlen = max(batch_lens)
    batch_x = [x + [0] * (maxlen-len(x)) for x in batch_x]
        
    batch_x = torch.tensor(batch_x, dtype=torch.int64, device=device)
    if batch_y is not None:
        batch_y = torch.tensor(batch_y, dtype=torch.int64, device=device)
    batch_lens = torch.tensor(batch_lens, dtype=torch.int64, device=device)
    return batch_x, batch_y, batch_lens
    

def sort_in_descending(batch_lens):
    '''
    Input: 
        batch_lens: tensor (int64)
    
    Return:
        sorted_batch_lens: tensor (int64)
        order: batch[order] -> sorted_batch
        revert_order: sorted_batch[revert_order] -> batch
    '''
    sorted_batch_lens, order = batch_lens.sort(descending=True)
    _, revert_order = order.sort(descending=False)
    return sorted_batch_lens, order, revert_order


def _test_sort_in_descending(batch_lens):
    sorted_batch_lens, order, revert_order = sort_in_descending(batch_lens)
    check = batch_lens[order] == sorted_batch_lens
    assert check.sum().item() == check.nelement()
    check = sorted_batch_lens[revert_order] == batch_lens
    assert check.sum().item() == check.nelement()
    

def norm_embedding_(emb_layer):
    '''Normalize each embedding vector. Very important. 
    '''
    emb_layer.weight.data.div_(emb_layer.weight.data.norm(2, dim=1, keepdim=True))
    
    
def init_embedding(gensim_w2v, dic):
    '''Return: tensor (float32)
    
    Updated: 20190430
    '''
    if not isinstance(gensim_w2v, tuple):
        vectors = gensim_w2v.wv.vectors
        index = gensim_w2v.wv.index2word
    else:
        vectors, word_sr = gensim_w2v
        index = word_sr.tolist()
        
    assert vectors.dtype == np.float32
    gensim_vecs = pd.DataFrame(vectors, index=index)
    new_index = [dic._idx2word[i] for i in range(dic.size)]
    gensim_vecs = gensim_vecs.reindex(new_index)
    
    sup_locs = gensim_vecs.iloc[:, 0].isna()
    gensim_vecs.loc[sup_locs] = np.random.uniform(-1, 1, (sup_locs.sum(), vectors.shape[1])).astype(np.float32)
    print('Substituted words by word2vec: %d/%d' % (dic.size - sup_locs.sum(), dic.size))
    
    emb = torch.from_numpy(gensim_vecs.values)
    emb.div_(emb.norm(2, dim=1, keepdim=True))
    return emb
    
    
def reinit_parameters_(layer, gain=1):
    '''Gain = 1 for tanh, gain = 4 for sigmoid.  
    '''
    for param in layer.parameters():
        if param.dim() == 1:
            param.data.fill_(0)
        elif param.dim() >= 2:
            nn.init.xavier_uniform_(param.data, gain=gain)
    
    
def reinit_parameters_lstm_(lstm_layer):
    '''
    W_i: (W_ii|W_if|W_ig|W_io) of shape (4*hidden_size x input_size)
    W_h: (W_hi|W_hf|W_hg|W_ho) of shape (4*hidden_size x hidden_size)
    W_{i, h}{i, f, o} use sigmoid activation function.
    W_{i, h}g use tanh activation function.
    
    http://deeplearning.net/tutorial/mlp.html#weight-initialization
    '''
    for param in lstm_layer.parameters():
        if param.dim() == 1:
            param.data.fill_(0)
        elif param.dim() == 2:
            n_hidden = param.size(0) // 4
            n_in = param.size(1)
            # Two inputs: hidden_{t-1} and input_{t}
            # One output: hidden_{t}
            lim_tanh = (6/(n_in + n_hidden*2)) ** 0.5
            lim_sigmoid = 4 * lim_tanh
            param.data.uniform_(-lim_sigmoid, lim_sigmoid)
            param.data[(2*n_hidden):(3*n_hidden)].div_(4)


def reinit_parameters_gru_(gru_layer):
    '''
    W_i: (W_ir|W_iz|W_in) of shape (3*hidden_size x input_size)
    W_h: (W_hr|W_hz|W_hn) of shape (3*hidden_size x hidden_size)
    W_{i, h}{r, z} use sigmoid activation function.
    W_{i, h}n use tanh activation function.
    
    http://deeplearning.net/tutorial/mlp.html#weight-initialization
    '''
    for param in gru_layer.parameters():
        if param.dim() == 1:
            param.data.fill_(0)
        elif param.dim() == 2:
            n_hidden = param.size(0) // 3
            n_in = param.size(1)
            # Two inputs: hidden_{t-1} and input_{t}
            # One output: hidden_{t}
            lim_tanh = (6/(n_in + n_hidden*2)) ** 0.5
            lim_sigmoid = 4 * lim_tanh
            param.data.uniform_(-lim_sigmoid, lim_sigmoid)
            param.data[(2*n_hidden):(3*n_hidden)].div_(4)
