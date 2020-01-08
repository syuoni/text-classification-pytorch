# -*- coding: utf-8 -*-
import os
import torch
import itertools
import pickle

from corpus import Corpus
from training import train_with_earlystop
    
    
if __name__ == '__main__':
    #TODO: cannot repeat results with same random-seed specified?
    n_hidden = 128
    n_emb = 128
    batch_size = 32
#    rng = np.random.RandomState(1224)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    use_hie = False
    w2v_fn = 'w2v/enwiki.w2v'
#    w2v_fn = None
    
#    dataset = 'imdb-2-200'
#    dataset = 'imdb'
#    dataset = 'yelp-2013'
    dataset = 'yelp-2014'
    dn = 'model-res-%s' % dataset
    if not os.path.exists(dn):
        os.makedirs(dn)
    
    corpus_fn = '%s/corpus-%s-with-cv.pkl' % (dn, dataset)
    if os.path.exists(corpus_fn):
        print('Corpus exists, loading...')
        with open(corpus_fn, 'rb') as f:
            corpus = pickle.load(f)
    else:
        print('Corpus creating...')
        corpus = Corpus.load_from_file('dataset/%s-prep.pkl' % dataset)
        corpus.create_cv(n_splits=5)
        with open(corpus_fn, 'wb') as f:
            pickle.dump(corpus, f)
    
#============================= ONE experiment ================================#
#    train_set, valid_set, test_set = corpus.train_valid_test()
#    train_with_validation(train_set, valid_set, use_w2v=False)
    
#============================ Cross Validation ===============================#
    for nn_type, pooling_type in itertools.product(['gru', 'lstm', 'conv'], ['mean', 'max', 'attention']):
#    for nn_type, pooling_type in itertools.product(['gru'], ['attention']):
        print(nn_type, pooling_type)
        
        for cv_idx in range(5):
            print('Dev fold: %d' % cv_idx)
            corpus.set_current_part(cv_idx)
            
            save_fn = '%s/model-%s-%s-%s-%d.ckpt' % (dn, nn_type, pooling_type, use_hie, cv_idx)
            if os.path.exists(save_fn):
                continue
            train_with_earlystop(corpus, device, n_hidden=n_hidden, n_emb=n_emb, batch_size=batch_size, 
                                 use_hie=use_hie, nn_type=nn_type, pooling_type=pooling_type, 
                                 w2v_fn=w2v_fn, save_fn=save_fn, disp_proc=False)
#            break
