# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn

from corpus import Corpus
from utils import cast_to_tensor
from models import construct_classifier
from models import FlatNNClassifier, HieNNClassifier


class Predictor(object):
    '''A conbination of a classifier and a corpus, which aims to predict 
    (batch of) w_seq directly. 
    WARNING: Must make sure the classifier and corpus are MATCHED!
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
        w_seq_df = pd.DataFrame({'w_seq': [w_seq]})
        w_seq_df['w_seq_len'] = w_seq_df['w_seq'].str.len()
        return self.decision_func_on_w_seq_df(w_seq_df, batch_size=1).flatten()
    
    def predict_on_w_seq(self, w_seq, with_prob=False):
        prob_distr = self.decision_func_on_w_seq(w_seq)
        pred_prob, predicted = prob_distr.max(dim=0)
        if with_prob:
            return (predicted, pred_prob)
        else:
            return predicted
    
        
class CVVotingPredictor(Predictor):
    '''A voting predictor constrcted from classifiers of a cross-validation. 
    WARNING: Must make sure the classifiers (loaded from classifier_state_paths) 
    and corpus (by cv partitions) are MATCHED one by one! 
    '''
    def __init__(self, classifier, classifier_state_paths, corpus):
        self.classifier = classifier
        self.classifier_state_paths = classifier_state_paths
        self.corpus = corpus
        
    def decision_func_on_w_seq_df(self, w_seq_df, batch_size=128):
        '''Use soft-voting
        '''
        prob_distr = 0
        for cv_idx, state_fn in enumerate(self.classifier_state_paths):
            print("Processing CV %d/%d..." % (cv_idx, len(self.classifier_state_paths)))
            self.corpus.set_current_part(cv_idx)
            
            # Resize the embeddings to match the current voc size
            self.classifier.voc_size = self.corpus.current_dic.size
            self.classifier.emb = nn.Embedding(self.classifier.voc_size, self.classifier.emb_dim)
            
            self.classifier.load_state_dict(torch.load(state_fn, map_location=None if torch.cuda.is_available() else 'cpu'))
            
            this_predictor = Predictor(self.classifier, self.corpus)
            prob_distr += this_predictor.decision_func_on_w_seq_df(w_seq_df, batch_size=batch_size)
            
        prob_distr = prob_distr / len(self.classifier_state_paths)
        return prob_distr
    


