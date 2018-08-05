# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec

from utils import init_embedding, cast_to_tensor
from models import NNClassifier


def train_batch(classifier, batch, loss_func, optimizer):
    batch_x, batch_y, batch_lens = cast_to_tensor(batch, device=classifier.device)
    
    classifier.zero_grad()
    cat_scores = classifier(batch_x, batch_lens)
    loss = loss_func(cat_scores, batch_y)
    loss.backward()
#    nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
    optimizer.step()
    return loss.item()
    
    
def train_batches(classifier, batches, loss_func, optimizer):
    losses = []
    for batch in batches:
        loss = train_batch(classifier, batch, loss_func, optimizer)
        losses.append(loss)
    return losses
    

def eval_batches(classifier, batches):
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
    

def train_with_earlystop(corpus, device, n_hidden=128, n_emb=128, batch_size=32, 
                         model_type='lstm', pooling='mean', bidirectional=True, conv_size=5, 
                         w2v_fn=None, save_fn=None, disp_proc=True):
    '''pooling_type: mean or max
    model_type: lstm, rnn or cnn
    use_w2v: whether to use pre-trained embeddings from word2vec
    '''
    print('%d training samples' % corpus.train_size)
    print('%d validation samples' % corpus.dev_size)
    
#    rng = np.random.RandomState(1224)
#    th_rng = RandomStreams(1224)
    
    emb2hidden = model_type
    if save_fn is None:
        save_fn = 'model-res/%s-%s.ckpt' % (emb2hidden, pooling)
    
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
    if emb2hidden == 'conv':
        classifier = NNClassifier(corpus.dic.size, n_emb, n_hidden, corpus.n_type, pre_embedding=pre_embedding, 
                                  emb2hidden=emb2hidden, pooling=pooling, 
                                  num_layers=1, conv_size=conv_size)
    else:
        classifier = NNClassifier(corpus.dic.size, n_emb, n_hidden, corpus.n_type, pre_embedding=pre_embedding, 
                                  emb2hidden=emb2hidden, pooling=pooling, 
                                  num_layers=1, bidirectional=bidirectional)
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
            
            if uidx % disp_freq == 0 and disp_proc:
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
    
    