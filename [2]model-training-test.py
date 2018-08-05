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
from models import HieNNClassifier, FlatNNClassifier
    
    
if __name__ == '__main__':
    #TODO: cannot repeat results with same random-seed specified?
    n_hidden = 256
    n_emb = 128
    batch_size = 32
    conv_size = 5
#    rng = np.random.RandomState(1224)
    
    dataset = 'imdb'
    corpus = Corpus.load_from_file('dataset/%s-prep.pkl' % dataset)
    corpus.create_cv(n_splits=5)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
#    nn_type = 'lstm'
    nn_type = 'gru'
#    nn_type = 'conv'
#    pooling = 'mean'
#    pooling = 'max'
    pooling = 'attention'

    bidirectional = True
#    w2v_fn = 'w2v/enwiki.w2v'
    w2v_fn = None
    save_fn = 'model-res/%s-%s.ckpt' % (nn_type, pooling)
    
#    if bidirectional:
#        n_hidden = n_hidden * 2
    
    # Load Word2Vec 
    if w2v_fn is None:
        pre_embedding = None
    else:
        print('Loading word2vec model...')
        if not os.path.exists(w2v_fn):
            raise Exception('Word2Vec model does NOT exist!', w2v_fn)
        gensim_w2v = Word2Vec.load(w2v_fn)
        pre_embedding = init_embedding(gensim_w2v, corpus.dic)
        
    # Define Model
    nn_kwargs = {'type': nn_type, 'num_layers': 1, 'bidirectional': True, 'conv_size': 5}
    pooling_kwargs = {'type': pooling, 'hidden_dim': n_hidden, 'atten_dim': n_hidden}
    emb2hidden_kwargs = {'nn_kwargs': nn_kwargs, 
                         'dropout_p': 0.5, 
                         'pooling_kwargs': pooling_kwargs}
    
    classifier = HieNNClassifier(corpus.dic.size, n_emb, n_hidden, corpus.n_type, pre_embedding=pre_embedding, 
                                 word2sent_kwargs=emb2hidden_kwargs, sent2doc_kwargs=emb2hidden_kwargs)
#    classifier = FlatNNClassifier(corpus.dic.size, n_emb, n_hidden, corpus.n_type, pre_embedding=pre_embedding, 
#                                  word2doc_kwargs=emb2hidden_kwargs)
    classifier.to(device)
    
    # Loss and Optimizer
    loss_func = nn.NLLLoss()
#    optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-8)
    optimizer = optim.Adagrad(classifier.parameters(), lr=0.01, weight_decay=1e-8)
    
    dev_batches  = list(corpus.iter_as_batches(batch_size=batch_size*5, shuffle=False, from_parts=['dev']))
    test_batches = list(corpus.iter_as_batches(batch_size=batch_size*5, shuffle=False, from_parts=['test']))
    
    # Train the model
    patience = 2500
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
        for batch in corpus.iter_as_batches(batch_size=batch_size, shuffle=True, from_parts=['train']):
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
