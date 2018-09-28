# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import reinit_parameters_, reinit_parameters_gru_, reinit_parameters_lstm_
from utils import sort_in_descending


def construct_nn_layer(emb_dim, hidden_dim, nn_type, nn_kwargs):
    if nn_type == 'gru':
        layer = GRULayer(emb_dim, hidden_dim, **nn_kwargs)
    elif nn_type == 'lstm':
        layer = LSTMLayer(emb_dim, hidden_dim, **nn_kwargs)
    elif nn_type == 'conv':
        layer = ConvLayer(emb_dim, hidden_dim, **nn_kwargs)
    else:
        raise Exception('Invalid NN type!', nn_type)
    return layer


class ConvLayer(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_layers=1, conv_size=5):
        super(ConvLayer, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        assert conv_size % 2 == 1
        self.conv_size = conv_size
        self.padding = (conv_size - 1) // 2
        
        for i in range(self.num_layers):
            if i == 0:
                this_conv = nn.Conv1d(emb_dim, hidden_dim, conv_size, padding=self.padding)
            else:
                this_conv = nn.Conv1d(hidden_dim, hidden_dim, conv_size, padding=self.padding)
            reinit_parameters_(this_conv)
            setattr(self, 'conv_%d' % i, this_conv)
        
    def forward(self, embed, batch_lens):
        '''
        Input:
            embed: tensor (float32) of shape (batch, step, emb)
            batch_lens: tensor (int64) of shape (batch, )
        Return:
            nn_outs: tensor (float32) of shape (batch, step, hidden)
        '''
        # RNN units use packed_sequence, which ensures padding areas with value 0, 
        # while using CNN units, it should be manually ensured...
        for batch_idx in range(embed.size(0)):
            embed[batch_idx, batch_lens[batch_idx].item():].fill_(0)
            
        # (batch, step, emb) -> (batch, emb, step)
        # The shape of hidden is (batch, emb, step) for now, 
        # and becomes (batch, hidden, step) after the first conv_layer. 
        hidden = embed.permute(0, 2, 1)
        
        # hidden: tensor (float32) of shape (batch, hidden, step)
        for i in range(self.num_layers):
            this_conv = getattr(self, 'conv_%d' % i)
            hidden = F.relu(this_conv(hidden))
        
        # (batch, hidden, step) -> (batch, step, hidden)
        # outs: tensor (float32) of shape (batch, step, hidden)
        outs = hidden.permute(0, 2, 1)
        
        # Fill the positions beyond valid lengths with zeros, consistent with 
        # the results of pad_packed_sequence
        for batch_idx in range(outs.size(0)):
            outs[batch_idx, batch_lens[batch_idx].item():].fill_(0)
        return outs
    
    
class GRULayer(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(GRULayer, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if bidirectional:
            assert hidden_dim % 2 == 0
            self.num_directions = 2
            self.hidden_dim_single = hidden_dim // 2
        else:
            self.num_directions = 1
            self.hidden_dim_single = hidden_dim
        
        # Trainable initial hidden state
        self.gru_h0 = nn.Parameter(torch.zeros(num_layers*self.num_directions, 1, self.hidden_dim_single), 
                                   requires_grad=True)
        # batch_first: (batch, seq, feature)
        self.gru = nn.GRU(emb_dim, self.hidden_dim_single, num_layers=num_layers, 
                          bidirectional=bidirectional, batch_first=True)
        reinit_parameters_gru_(self.gru)
        
    def forward(self, embed, batch_lens):
        '''
        Input:
            embed: tensor (float32) of shape (batch, step, emb)
            batch_lens: tensor (int64) of shape (batch, )
        Return:
            nn_outs: tensor (float32) of shape (batch, step, hidden)
        '''
        # Sort in a descending order
        sorted_batch_lens, order, revert_order = sort_in_descending(batch_lens)
        sorted_embed = embed[order]
        packed = pack_padded_sequence(sorted_embed, lengths=sorted_batch_lens, batch_first=True)
        
        # NOTE: CUDNN_STATUS_EXECUTION_FAILED if device is not appropriately specified     
        h0 = self.gru_h0.repeat(1, sorted_embed.size(0), 1)
        packed_outs, h = self.gru(packed, h0)
        # outs: tensor (float32) of shape (batch, step, hidden)
        sorted_outs, _ = pad_packed_sequence(packed_outs, batch_first=True)
        
        # Revert to the original order
        return sorted_outs[revert_order]
    
    
class LSTMLayer(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(LSTMLayer, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if bidirectional:
            assert hidden_dim % 2 == 0
            self.num_directions = 2
            self.hidden_dim_single = hidden_dim // 2
        else:
            self.num_directions = 1
            self.hidden_dim_single = hidden_dim
        
        # Trainable initial hidden state
        self.lstm_h0 = nn.Parameter(torch.zeros(num_layers*self.num_directions, 1, self.hidden_dim_single), 
                                    requires_grad=True)
        self.lstm_c0 = nn.Parameter(torch.zeros(num_layers*self.num_directions, 1, self.hidden_dim_single), 
                                    requires_grad=True)
        # batch_first: (batch, seq, feature)
        self.lstm = nn.LSTM(emb_dim, self.hidden_dim_single, num_layers=num_layers, 
                            bidirectional=bidirectional, batch_first=True)
        reinit_parameters_lstm_(self.lstm)        
        
    def forward(self, embed, batch_lens):
        '''
        Input:
            embed: tensor (float32) of shape (batch, step, emb)
            batch_lens: tensor (int64) of shape (batch, )
        Return:
            nn_outs: tensor (float32) of shape (batch, step, hidden)
        '''
        # Sort in a descending order
        sorted_batch_lens, order, revert_order = sort_in_descending(batch_lens)
        sorted_embed = embed[order]
        packed = pack_padded_sequence(sorted_embed, lengths=sorted_batch_lens, batch_first=True)
        
        # NOTE: CUDNN_STATUS_EXECUTION_FAILED if device is not appropriately specified     
        h0 = self.lstm_h0.repeat(1, sorted_embed.size(0), 1)
        c0 = self.lstm_c0.repeat(1, sorted_embed.size(0), 1)
        packed_outs, (h, c) = self.lstm(packed, (h0, c0))
        # outs: tensor (float32) of shape (batch, step, hidden)
        sorted_outs, _ = pad_packed_sequence(packed_outs, batch_first=True)
        
        # Revert to the original order
        return sorted_outs[revert_order]   
    