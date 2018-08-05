# -*- coding: utf-8 -*-
import numpy as np
import torch

class Predictor(object):
    '''A conbination of classifier, tokenizer and dictionary, 
    which aims to predict (batch of) sentences directly. 
    '''
    def __init__(self, classifier, tokenizer, dic):
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.dic = dic
    
    def predict_sent(self, sent, with_prob=False):
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