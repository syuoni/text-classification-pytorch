# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec

from corpus import Corpus
from training import train_batch, eval_batches
from utils import init_embedding
from models import construct_classifier
    
    
if __name__ == '__main__':
#    for lr in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
#    for lr in [0.0005, 0.0002, 0.0001]:
    #TODO: cannot repeat results with same random-seed specified?
    n_hidden = 128
    n_emb = 128
    batch_size = 32
#    rng = np.random.RandomState(1224)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    use_hie = False
    w2v_fn = 'w2v/enwiki.w2v'
#    w2v_fn = None
    
#    dataset = 'imdb'
    dataset = 'yelp-2013'
#    dataset = 'yelp-2014'
    corpus = Corpus.load_from_file('dataset/%s-prep.pkl' % dataset)
    
    
    nn_type = 'gru'
#    nn_type = 'lstm'
#    nn_type = 'conv'
#    pooling_type = 'mean'
#    pooling_type = 'max'
    pooling_type = 'attention'
    
    # Load Word2Vec 
    if w2v_fn is None:
        pre_embedding = None
    else:
        print('Loading word2vec model...')
        if not os.path.exists(w2v_fn):
            raise Exception('Word2Vec model does NOT exist!', w2v_fn)
        gensim_w2v = Word2Vec.load(w2v_fn)
        pre_embedding = init_embedding(gensim_w2v, corpus.current_dic)
        
    classifier = construct_classifier(corpus.current_dic.size, n_emb, n_hidden, corpus.n_target, 
                                      pre_embedding=pre_embedding, use_hie=use_hie, 
                                      nn_type=nn_type, pooling_type=pooling_type)
#    classifier.load_state_dict(torch.load('model-res/model-gru-attention-False-Adadelta-1.0000-v1.ckpt'))
    classifier.to(device)
    
    # Loss and Optimizer
    loss_func = nn.NLLLoss()
    # It seems that Adadelta is better than Adagrad and Adam...
    
#    optim_type = 'SGD'
    optim_type = 'Adadelta'
    lr = 1.0
    if optim_type == 'SGD':
        optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=1e-8)
    elif optim_type == 'Adadelta':
        optimizer = optim.Adadelta(classifier.parameters(), lr=lr, rho=0.95, weight_decay=1e-8)
    else:
        raise Exception('Invalid optimizer!', optim_type)
    

#    optimizer = optim.Adagrad(classifier.parameters(), lr=0.01, weight_decay=1e-8)
#    optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-8)
#    optimizer = optim.Adamax(classifier.parameters(), lr=0.001, weight_decay=1e-8)
    
    save_fn = 'model-res/model-%s-%s-%s-%s-%.4f-v3.ckpt' % (nn_type, pooling_type, use_hie, optim_type, lr)
    
    
    dev_batches  = list(corpus.iter_as_batches(batch_size=batch_size, order='descending', from_parts=['dev']))
    test_batches = list(corpus.iter_as_batches(batch_size=batch_size, order='descending', from_parts=['test']))
    
    # Train the model
#    patience = 2500
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    disp_freq = 20
    validation_freq = 200
    
    max_epoch = 500
    best_iter = 0
    best_validation_err = 1.0
    
    epoch = 0
    uidx = 0
    done_looping = False
    start_time = time.time()
    
    
    while (epoch < max_epoch) and (not done_looping):
        epoch += 1
        # Get new shuffled batches from training set. 
        for batch in corpus.iter_as_batches(batch_size=batch_size, order='shuffle', from_parts=['train']):
            uidx += 1
            train_loss = train_batch(classifier, batch, loss_func, optimizer)
            
            if uidx % disp_freq == 0:
                print('epoch %i, minibatch %i, train loss %f' % (epoch, uidx, train_loss))
                
            if uidx % validation_freq == 0:
                this_validation_err = eval_batches(classifier, dev_batches)
                print('epoch %i, minibatch %i, validation error %f %%' % (epoch, uidx, this_validation_err*100))
                
                if this_validation_err < best_validation_err:
                    if this_validation_err < best_validation_err*improvement_threshold:
                        patience = max(patience, uidx*patience_increase)
                        
                    best_validation_err = this_validation_err
                    best_iter = uidx
                    test_err = eval_batches(classifier, test_batches)
                    print('    epoch %i, minibatch %i, test error %f %%' % (epoch, uidx, test_err*100))
                    
                    torch.save(classifier.state_dict(), save_fn)
                    # classifier.load_state_dict(torch.load(save_fn))
                    
            if patience < uidx:
                done_looping = True
                break
        
    end_time = time.time()
    print('Optimization complete with best validation score of %f %%, at iter %d, with test performance %f %%' % (best_validation_err * 100, best_iter, test_err * 100))
    print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time)))
