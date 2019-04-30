# -*- coding: utf-8 -*-
import numpy as np
import torch
import pickle

from corpus import Corpus
from utils import cast_to_tensor
from models import construct_classifier
from models import FlatNNClassifier, HieNNClassifier


class Predictor(object):
    '''A conbination of a classifier and a corpus, 
    which aims to predict (batch of) w_seq directly. 
    '''
    def __init__(self, classifier, corpus):
        self.classifier = classifier
        self.corpus = corpus
        
    def decision_func_on_w_seq_df(self, w_seq_df, batch_size=128):
        batches = list(self.corpus.iter_as_batches(batch_size=batch_size, order='original', input_df=w_seq_df))
        # Check if keep the original order
        assert all([x==y for x, y in zip(batches[0][-1], w_seq_df['w_seq_len'].iloc[:batch_size].tolist())])
        
        self.classifier.eval()
        with torch.no_grad():
            prob_distr = []
            for batch_idx, batch in enumerate(batches):
                batch_x, batch_y, batch_lens = cast_to_tensor(batch, device=self.classifier.device)
                # Shape (batch_size, n_targets), probability distribution
                this_prob_distr = self.classifier.decision_func(batch_x, batch_lens)
                prob_distr.append(this_prob_distr)
                if (batch_idx+1) % 100 == 0:
                    print('Processing %d/%d...' % (batch_idx+1, len(batches)))
                
        self.classifier.train()
        prob_distr = torch.cat(prob_distr, dim=0)
        return prob_distr
    
    def predict_on_w_seq_df(self, w_seq_df, batch_size=128, with_prob=False):
        prob_distr = self.decision_func_on_w_seq_df(w_seq_df, batch_size=batch_size)
        pred_prob, predicted = prob_distr.max(dim=1)
        if with_prob:
            return (predicted, pred_prob)
        else:
            return predicted
        
    def decision_func_on_w_seq(self, w_seq):
        pass
    
    def predict_on_w_seq(self, w_seq):
        pass
    
        
#        if self.voting == 'hard':
#            # sub_res -> (estimator_dim, )
#            sub_res = np.array([estimator.predict_sent(sent) for estimator in self.estimators], 
#                               dtype=np.float32)
#            mode_res, count = mode(sub_res)
#            return (mode_res[0], count[0]/self.n_estimators) if with_prob else mode_res[0]
#        else:
#            # sub_res -> (estimator_dim, target_dim)
#            sub_res = np.array([estimator.predict_sent(sent, with_prob=True) for estimator in self.estimators], 
#                               dtype=np.float32)
#            sub_res = sub_res.mean(axis=0)
#            max_res = np.argmax(sub_res)
#            mean_prob = sub_res[max_res]
#            return (max_res, mean_prob) if with_prob else max_res
    
#    def predict(self, batch_x, batch_lens, with_prob=False):
#        cat_scores = self.decision_func(batch_x, batch_lens)
#        pred_prob, predicted = cat_scores.max(dim=1)
#        if with_prob:
#            return (predicted, pred_prob)
#        else:
#            return predicted
#    
#    def decision_func(self, batch_x, batch_lens):
#        # sub_res: (classifier_dim, batch_dim, target_dim)
#        sub_res = torch.stack([classifier.decision_func(batch_x, batch_lens) for classifier in self.classifiers])
#        return sub_res.mean(dim=0)
        
    


