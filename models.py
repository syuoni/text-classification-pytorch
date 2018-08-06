# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import norm_embedding_, reinit_parameters_
from nn_layers import construct_nn_layer
from pooling_layers import construct_pooling_layer


class NNClassifier(nn.Module):
    '''
    Flat structure: 
        x -> Embedding -> embed
        embed -> word2doc (NN + Dropout + Pooling) -> doc_hidden
        doc_hidden -> FC -> cat
    Hierarchical structure:
        x -> Embedding -> embed
        embed -> word2sent (NN + Dropout + Pooling) -> sent_hidden
        sent_hidden -> sent2doc (NN + Dropout + Pooling) -> doc_hidden
        doc_hidden -> FC -> cat
    '''
    def __init__(self, voc_size, emb_dim, hidden_dim, cat_size, pre_embedding=None):
        super(NNClassifier, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.cat_size = cat_size
        self.device = torch.device('cpu')
        
        if pre_embedding is None:
            self.emb = nn.Embedding(voc_size, emb_dim)
            norm_embedding_(self.emb)
        else:
            assert voc_size == pre_embedding.size(0)
            assert emb_dim == pre_embedding.size(1)
            self.emb = nn.Embedding.from_pretrained(pre_embedding, freeze=False)
        
        self.hidden2cat = nn.Linear(hidden_dim, cat_size)
        reinit_parameters_(self.hidden2cat)        
        
    def forward(self, batch_x, batch_lens):
        '''
        Input:
            batch_x: tensor (int64) of shape (batch, step)
            batch_lens: tensor (int64) of shape (batch, )
        Return:
            cat_scores: tensor (float32) of shape (batch, category)
        '''        
        # embed: tensor (float32) of shape (batch, step, emb)
        word_embed = self.emb(batch_x)
        
        # doc_embed: tensor (float32) of shape (batch, hidden)
        doc_embed = self._forward_word2doc(word_embed, batch_x, batch_lens)
        
        # cat_space: tensor (float32) of shape (batch, category)
        cat_space = self.hidden2cat(doc_embed)
        cat_scores = F.log_softmax(cat_space, dim=-1)
        return cat_scores
    
    def _forward_word2doc(self, word_embed, batch_x, batch_lens):
        '''
        Input:
            word_embed: tensor (float32) of shape (batch, step, emb)
            batch_x: tensor (int64) of shape (batch, step)
            batch_lens: tensor (int64) of shape (batch, )
        Return:
            doc_embed: tensor (float32) of shape (batch, hidden)
        '''
        raise NotImplementedError
        
    def predict(self, batch_x, batch_lens, with_prob=False):
        cat_scores = self.forward(batch_x, batch_lens)
        pred_prob, predicted = cat_scores.max(dim=1)
        if with_prob:
            return (predicted, pred_prob.exp())
        else:
            return predicted
        
    def decision_func(self, batch_x, batch_lens):
        '''
        Return: tensor (float32) of shape (batch, cat_size)
        '''
        cat_scores = self.forward(batch_x, batch_lens)
        return cat_scores.exp()
    
    def to(self, device):
        super(NNClassifier, self).to(device)
        self.device = device
    

class FlatNNClassifier(NNClassifier):
    '''
    Flat structure: 
        x -> Embedding -> embed
        embed -> word2doc (NN + Dropout + Pooling) -> doc_hidden
        doc_hidden -> FC -> cat
    '''    
    def __init__(self, voc_size, emb_dim, hidden_dim, cat_size, pre_embedding=None, word2doc_info=None):
        super(FlatNNClassifier, self).__init__(voc_size, emb_dim, hidden_dim, cat_size, pre_embedding=pre_embedding)
        self.word2doc_nn = construct_nn_layer(emb_dim, hidden_dim, word2doc_info['nn_type'], word2doc_info['nn_kwargs'])
        self.word2doc_dropout = nn.Dropout(word2doc_info['dropout_p'])
        self.word2doc_pooling = construct_pooling_layer(word2doc_info['pooling_type'], word2doc_info['pooling_kwargs'])
        
    def _forward_word2doc(self, word_embed, batch_x, batch_lens):
        '''
        Input:
            word_embed: tensor (float32) of shape (batch, step, emb)
            batch_x: tensor (int64) of shape (batch, step)
            batch_lens: tensor (int64) of shape (batch, )
        Return:
            doc_embed: tensor (float32) of shape (batch, hidden)
        '''
        # word_hidden: tensor (float32) of shape (batch, step, hidden)
        word_hidden = self.word2doc_nn(word_embed, batch_lens)
        word_hidden = self.word2doc_dropout(word_hidden)
        
        # doc_embed: tensor (float32) of shape (batch, hidden)
        doc_embed = self.word2doc_pooling(word_hidden, batch_lens)        
        return doc_embed
        
    
class HieNNClassifier(NNClassifier):
    '''
    Hierarchical structure:
        x -> Embedding -> embed
        embed -> word2sent (NN + Dropout + Pooling) -> sent_hidden
        sent_hidden -> sent2doc (NN + Dropout + Pooling) -> doc_hidden
        doc_hidden -> FC -> cat
    '''
    def __init__(self, voc_size, emb_dim, hidden_dim, cat_size, pre_embedding=None, 
                 word2sent_info=None, sent2doc_info=None, state_pass=False):
        '''
        Input:
            state_pass: whether hidden states pass through sentences in the word2sent. 
        '''
        super(HieNNClassifier, self).__init__(voc_size, emb_dim, hidden_dim, cat_size, pre_embedding=pre_embedding)
        self.state_pass = state_pass
        
        word2sent_nn = construct_nn_layer(emb_dim, hidden_dim, word2sent_info['nn_type'], word2sent_info['nn_kwargs'])
        word2sent_dropout = nn.Dropout(word2sent_info['dropout_p'])
        word2sent_pooling = construct_pooling_layer(word2sent_info['pooling_type'], word2sent_info['pooling_kwargs'])
        
        if self.state_pass:
            self.word2sent_nn = word2sent_nn
            self.word2sent_dropout = word2sent_dropout
            self.word2sent_hie = HieLayer(word2sent_pooling)
        else:
            self.word2sent_hie = HieLayer(word2sent_pooling, nn_layer=word2sent_nn, dropout_layer=word2sent_dropout)
        
        self.sent2doc_nn = construct_nn_layer(hidden_dim, hidden_dim, sent2doc_info['nn_type'], sent2doc_info['nn_kwargs'])
        self.sent2doc_dropout = nn.Dropout(sent2doc_info['dropout_p'])
        self.sent2doc_pooling = construct_pooling_layer(sent2doc_info['pooling_type'], sent2doc_info['pooling_kwargs'])
        
    def _forward_word2doc(self, word_embed, batch_x, batch_lens):
        '''
        Input:
            word_embed: tensor (float32) of shape (batch, step, emb)
            batch_x: tensor (int64) of shape (batch, step)
            batch_lens: tensor (int64) of shape (batch, )
        Return:
            doc_embed: tensor (float32) of shape (batch, hidden)
        '''
        if self.state_pass:
            # word_hidden: tensor (float32) of shape (batch, step, hidden)
            word_hidden = self.word2sent_nn(word_embed, batch_lens)
            word_hidden = self.word2sent_dropout(word_hidden)
            
            # sent_embed: tensor (float32) of shape (batch, sent_step, hidden)
            # batch_agg_lens: tensor (int64) of shape (batch, )
            # batch_lens indicates number of words in documents, 
            # while batch_agg_lens indicates number of sentences in documents. 
            sent_embed, batch_agg_lens = self.word2sent_hie(word_hidden, batch_x)
        else:
            sent_embed, batch_agg_lens = self.word2sent_hie(word_embed, batch_x)
        
        # sent_hidden: tensor (float32) of shape (batch, sent_step, hidden)
        sent_hidden = self.sent2doc_nn(sent_embed, batch_agg_lens)
        sent_hidden = self.sent2doc_dropout(sent_hidden)
        
        # doc_embed: tensor (float32) of shape (batch, hidden)
        doc_embed = self.sent2doc_pooling(sent_hidden, batch_agg_lens)
        return doc_embed
    
    
class HieLayer(nn.Module):
    def __init__(self, pooling_layer, nn_layer=None, dropout_layer=None):
        super(HieLayer, self).__init__()
        self.nn_layer = nn_layer
        self.dropout_layer = dropout_layer
        self.pooling_layer = pooling_layer
            
    def forward(self, hie_ins, batch_x):
        '''
        Input:
            hie_ins: tensor (float32) of shape (batch, step, emb/hidden)
            batch_x: tensor (int64) of shape (batch, step)
        Return:
            doc_list: tensor (float32) of shape (batch, sent_step_in_doc, hidden)
            doc_lens: tensor (int64) of shape (batch, )
        '''
        device = batch_x.device
        doc_lens = []  # Number of sentences in documents
        sent_lens = [] # Number of words in sentence
        sent_list = [] 
        
        for batch_idx in range(batch_x.size(0)):
            cuts = torch.arange(batch_x.size(1), dtype=torch.int64, device=device)[batch_x[batch_idx]==1] + 1
            cuts = cuts.cpu().numpy().tolist()
            doc_lens.append(len(cuts))
            
            for cut0, cut1 in zip([0] + cuts, cuts):
                sent_list.append(hie_ins[batch_idx, cut0:cut1])
                sent_lens.append(cut1 - cut0)
        
        doc_maxlen = max(doc_lens)
        sent_maxlen = max(sent_lens)
        
        # sent_list: tensor (float32) of shape (n_sent, word_step_in_sent, emb/hidden)
        sent_list = [F.pad(sent, (0, 0, 0, sent_maxlen-sent_len)) for sent, sent_len in zip(sent_list, sent_lens)]
        sent_list = torch.stack(sent_list)
        
        # sent_list: tensor (float32) of shape (n_sent, word_step_in_sent, hidden)
        if self.nn_layer is not None:
            sent_list = self.nn_layer(sent_list, torch.tensor(sent_lens, dtype=torch.int64, device=device))
        if self.dropout_layer is not None:
            sent_list = self.dropout_layer(sent_list)
        
        # sent_pooled: tensor (float32) of shape (n_sent, hidden)
        sent_pooled = self.pooling_layer(sent_list, torch.tensor(sent_lens, dtype=torch.int64, device=device))
        
        # doc_list: tensor (float32) of shape (batch, sent_step_in_doc, hidden)
        doc_cuts = np.cumsum(doc_lens).tolist()
        doc_list = [sent_pooled[cut0:cut1] for cut0, cut1 in zip([0] + doc_cuts, doc_cuts)]
        doc_list = [F.pad(doc, (0, 0, 0, doc_maxlen-doc_len)) for doc, doc_len in zip(doc_list, doc_lens)]
        doc_list = torch.stack(doc_list)
        
        return doc_list, torch.tensor(doc_lens, dtype=torch.int64, device=device)
        
#        interm_hidden_list = []
#        interm_batch_lens = []        
#        for batch_idx in range(batch_x.size(0)):  
#            cuts = torch.arange(batch_x.size(1), dtype=torch.int64, device=batch_x.device)[batch_x[batch_idx]==1] + 1
#            
#            sub_pooled_outs_list = []
#            for cut_idx in range(cuts.size(0)):
#                cut0 = 0 if cut_idx == 0 else cuts[cut_idx-1]
##                print(batch_x[batch_idx, cut0:cuts[cut_idx]][-1])
#                
#                # sub_outs: tensor (float32) of shape (sub_step, hidden)
#                sub_outs = nn_outs[batch_idx, cut0:cuts[cut_idx]]
#                # sub_pooled_outs: tensor (float32) of shape (1, hidden)
#                sub_pooled_outs = self.pooling(sub_outs.unsqueeze(0), 
#                                               torch.tensor([sub_outs.size(0)], device=batch_x.device))
#                sub_pooled_outs_list.append(sub_pooled_outs)
#                
##            import pdb; pdb.set_trace()
#            interm_hidden = torch.cat(sub_pooled_outs_list)
#            interm_hidden_list.append(interm_hidden)
#            interm_batch_lens.append(interm_hidden.size(0))
#            
#        maxlen = max(interm_batch_lens)
#        interm_hidden_list = [F.pad(interm_hidden_list[i], (0, 0, 0, maxlen-interm_batch_lens[i])) for i in range(batch_x.size(0))]
#        
##        for x in interm_hidden_list:
##            print(x.size())
#        
#        return torch.stack(interm_hidden_list), torch.tensor(interm_batch_lens, device=batch_x.device)
        
    
class VotingClassifier(object):
    def __init__(self, sub_classifiers):
        self.classifiers = sub_classifiers
        self.n_classifiers = len(sub_classifiers)
        self.device = sub_classifiers[0].device
        
    def predict(self, batch_x, batch_lens, with_prob=False):
        cat_scores = self.decision_func(batch_x, batch_lens)
        pred_prob, predicted = cat_scores.max(dim=1)
        if with_prob:
            return (predicted, pred_prob)
        else:
            return predicted
    
    def decision_func(self, batch_x, batch_lens):
        # sub_res: (classifier_dim, batch_dim, target_dim)
        sub_res = torch.stack([classifier.decision_func(batch_x, batch_lens) for classifier in self.classifiers])
        return sub_res.mean(dim=0)
        
    def eval(self):
        for classifier in self.classifiers:
            classifier.eval()
            
    def train(self):
        for classifier in self.classifiers:
            classifier.train()
        
    def to(self, device):
        for classifier in self.classifiers:
            classifier.to(device)
        self.device = device
        
        