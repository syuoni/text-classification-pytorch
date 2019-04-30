# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec

from utils import init_embedding, cast_to_tensor
from models import construct_classifier


def train_batch(classifier, batch, loss_func, optimizer):
    batch_x, batch_y, batch_lens = cast_to_tensor(batch, device=classifier.device)
    
    classifier.zero_grad()
    cat_scores = classifier(batch_x, batch_lens)
    loss = loss_func(cat_scores, batch_y)
    loss.backward()
    # Clip the gradients
    nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
    optimizer.step()
    return loss.item()
    
    
def train_batches(classifier, batches, loss_func, optimizer):
    losses = []
    for batch in batches:
        loss = train_batch(classifier, batch, loss_func, optimizer)
        losses.append(loss)
    return losses
    

def eval_batches(classifier, batches):
    '''Accept VotingClassifier, but need to ensure the batches are consistent to
    all sub-classifiers. 
    '''
    classifier.eval()
    with torch.no_grad():
        n_sample = 0
        n_error = 0
        for batch in batches:
            batch_x, batch_y, batch_lens = cast_to_tensor(batch, device=classifier.device)
            n_sample += batch_x.size(0)
            n_error += (classifier.predict(batch_x, batch_lens) != batch_y).sum().item()
            
    classifier.train()
    return n_error / n_sample
    

def eval_corpus(classifier, corpus, batch_size=32):
    
    pass



def train_corpus(classifier, corpus, loss_func, optimizer, save_fn, 
                 batch_size=32, patience=5000, valid_freq=100, disp_freq=20):
    # Fixed development batches
    dev_batches  = list(corpus.iter_as_batches(batch_size=batch_size, order='descending', from_parts=['dev']))
    
    # Additional configuration
    patience_increase = 2
    improvement_threshold = 0.995
    
    # Initial validation
    best_iter = 0
    best_valid_err = eval_batches(classifier, dev_batches)
    print('Initial validation error %f %%' % (best_valid_err*100))
    torch.save(classifier.state_dict(), save_fn)
    
    # Train the model
    max_epoch = 500
    epoch = 0
    uidx = 0
    done_looping = False
    start_time = time.time()
    
    while (epoch < max_epoch) and (not done_looping):
        epoch += 1
        # Get new shuffled batches from training set
        for batch in corpus.iter_as_batches(batch_size=batch_size, order='shuffle', from_parts=['train']):
            uidx += 1
            train_loss = train_batch(classifier, batch, loss_func, optimizer)
            
            # DO NOT display if disp_freq is None
            if disp_freq is not None and uidx % disp_freq == 0:
                print('Epoch %i, minibatch %i, train loss %f' % (epoch, uidx, train_loss))
    
            if uidx % valid_freq == 0:
                this_valid_err = eval_batches(classifier, dev_batches)
                print('Epoch %i, minibatch %i, validation error %f %%' % (epoch, uidx, this_valid_err*100))
                
                if this_valid_err < best_valid_err:
                    if this_valid_err < best_valid_err*improvement_threshold:
                        patience = max(patience, uidx*patience_increase)
                        
                    best_valid_err = this_valid_err
                    best_iter = uidx
                    torch.save(classifier.state_dict(), save_fn)
                    
            if patience < uidx:
                done_looping = True
                break
        
    end_time = time.time()
    print('Optimization complete with best validation score of %f %%, at iter %d' % (best_valid_err * 100, best_iter))
    print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time)))


def train_with_earlystop(corpus, device, n_hidden=128, n_emb=128, batch_size=32, 
                         use_hie=False, nn_type='gru', pooling_type='attention', 
                         w2v_fn=None, save_fn=None, disp_proc=True):
    '''
    Input:
        use_hie: whether use hierarchical structure. 
        nn_type: gru, lstm, conv
        pooling_type: mean, max, attention
        use_w2v: whether to use pre-trained embeddings from word2vec
    '''
    print('%d training samples' % corpus.current_split_sizes[0])
    print('%d validation samples' % corpus.current_split_sizes[1])
    
#    rng = np.random.RandomState(1224)
#    th_rng = RandomStreams(1224)
    
    if save_fn is None:
        save_fn = 'model-res/%s-%s-%s.ckpt' % (nn_type, pooling_type, use_hie)
    
    # Load Word2Vec 
    if w2v_fn is None:
        pre_embedding = None
    else:
        print('Loading word2vec model...')
        if w2v_fn == 'tencent':
            vectors = np.load(r'G:\word2vec\Tencent-AI-Lab\tencent-ailab-vecs-128.npy')
            word_sr = pd.read_hdf(r'G:\word2vec\Tencent-AI-Lab\tencent-ailab-voc.h5', 'voc')
            gensim_w2v = (vectors, word_sr)
        else:
            gensim_w2v = Word2Vec.load(w2v_fn)
        pre_embedding = init_embedding(gensim_w2v, corpus.current_dic)
        
    classifier = construct_classifier(corpus.current_dic.size, n_emb, n_hidden, corpus.n_target, 
                                      pre_embedding=pre_embedding, use_hie=use_hie, 
                                      nn_type=nn_type, pooling_type=pooling_type)
    classifier.to(device)
    
    # Loss and Optimizer
    loss_func = nn.NLLLoss()
    adadelta_optimizer = optim.Adadelta(classifier.parameters(), lr=1.0, rho=0.9, weight_decay=1e-8)
    sgd_optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-8)
    
    # First step: optimize using Adadelta
    disp_freq = 20 if disp_proc is True else None
    train_corpus(classifier, corpus, loss_func, adadelta_optimizer, save_fn, 
                 disp_freq=disp_freq, batch_size=batch_size)
    
    # Retrieve the state optimized by Adadelta
    classifier.load_state_dict(torch.load(save_fn))
    # Second step: optimize using SGD
    train_corpus(classifier, corpus, loss_func, sgd_optimizer, save_fn, 
                 disp_freq=disp_freq, batch_size=batch_size)
    
    
    
    
    
    