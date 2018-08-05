# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import norm_embedding_, reinit_parameters_
from nn_layers import construct_nn_layer
from pooling_layers import construct_pooling_layer, HiePoolingLayer


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
        doc_embed = self._forward_emb2doc(word_embed, batch_x, batch_lens)
        
        # cat_space: tensor (float32) of shape (batch, category)
        cat_space = self.hidden2cat(doc_embed)
        cat_scores = F.log_softmax(cat_space, dim=-1)
        return cat_scores
    
    def _forward_emb2doc(self, word_embed, batch_x, batch_lens):
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
    def __init__(self, voc_size, emb_dim, hidden_dim, cat_size, pre_embedding=None, word2doc_kwargs=None):
        super(FlatNNClassifier, self).__init__(voc_size, emb_dim, hidden_dim, cat_size, pre_embedding=pre_embedding)
        self.word2doc_nn = construct_nn_layer(emb_dim, hidden_dim, word2doc_kwargs['nn_kwargs'])
        self.word2doc_dropout = nn.Dropout(word2doc_kwargs['dropout_p'])
        self.word2doc_pooling = construct_pooling_layer(word2doc_kwargs['pooling_kwargs'])
        
    def _forward_emb2doc(self, word_embed, batch_x, batch_lens):
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
                 word2sent_kwargs=None, sent2doc_kwargs=None):
        super(HieNNClassifier, self).__init__(voc_size, emb_dim, hidden_dim, cat_size, pre_embedding=pre_embedding)
        self.word2sent_nn = construct_nn_layer(emb_dim, hidden_dim, word2sent_kwargs['nn_kwargs'])
        self.word2sent_dropout = nn.Dropout(word2sent_kwargs['dropout_p'])
        word2sent_pooling = construct_pooling_layer(word2sent_kwargs['pooling_kwargs'])
        self.word2sent_hie_pooling = HiePoolingLayer(word2sent_pooling)
        
        self.sent2doc_nn = construct_nn_layer(hidden_dim, hidden_dim, sent2doc_kwargs['nn_kwargs'])
        self.sent2doc_dropout = nn.Dropout(sent2doc_kwargs['dropout_p'])
        self.sent2doc_pooling = construct_pooling_layer(sent2doc_kwargs['pooling_kwargs'])
        
    def _forward_emb2doc(self, word_embed, batch_x, batch_lens):
        '''
        Input:
            word_embed: tensor (float32) of shape (batch, step, emb)
            batch_x: tensor (int64) of shape (batch, step)
            batch_lens: tensor (int64) of shape (batch, )
        Return:
            doc_embed: tensor (float32) of shape (batch, hidden)
        '''
        # word_hidden: tensor (float32) of shape (batch, step, hidden)
        word_hidden = self.word2sent_nn(word_embed, batch_lens)
        word_hidden = self.word2sent_dropout(word_hidden)
        
        # sent_embed: tensor (float32) of shape (batch, sent_step, hidden)
        # batch_sent_lens: tensor (int64) of shape (batch, )
        sent_embed, batch_sent_lens = self.word2sent_hie_pooling(word_hidden, batch_x)
        
        # sent_hidden: tensor (float32) of shape (batch, sent_step, hidden)
        sent_hidden = self.sent2doc_nn(sent_embed, batch_sent_lens)
        sent_hidden = self.sent2doc_dropout(sent_hidden)
        
        # doc_embed: tensor (float32) of shape (batch, hidden)
        doc_embed = self.sent2doc_pooling(word_hidden, batch_sent_lens)
        return doc_embed
    
    
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
        
        