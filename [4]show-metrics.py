# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch

from corpus import Corpus
from models import FlatNNClassifier, HieNNClassifier, VotingClassifier
from training import eval_batches


n_hidden = 256
n_emb = 128
batch_size = 64
conv_size = 5
bidirectional = True

dataset = 'imdb'
dn = 'model-res-%s' % dataset
corpus = Corpus.load_from_file('%s/%s-with-cv.pkl' % (dn, dataset))
test_batches = list(corpus.iter_as_batches(batch_size=batch_size*5, shuffle=False, from_parts=['test']))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#=========================== Metrics Calculation =============================#
'''
|Model| Pooling |IMDB(2)|IMDB(10)|Yelp 2013|Yelp 2014|
|:---:|:-------:|:-----:|:------:|:-------:|:-------:|
|GRNN |Mean     | 0.9121|  0.4691| | |
|GRNN |Max      | 0.9246|  0.4781| | |
|GRNN |Attention| 0.9259|  0.4855| | |
|LSTM |Mean     | 0.9068|  0.3855| | |
|LSTM |Max      | 0.9234|  0.4836| | |
|LSTM |Attention| 0.9233|  0.4813| | |
|CNN  |Mean     | 0.9077|  0.4011| | |
|CNN  |Max      | 0.9204|  0.4884| | |
|CNN  |Attention| 0.9198|  0.4795| | |
'''
#=============================================================================#
use_hie = False


#nn_type = 'gru'
nn_type = 'lstm'
#nn_type = 'conv'
pooling_type = 'mean'
#pooling_type = 'max'
#pooling_type = 'attention'

save_dn = '%s/%s-%s-%s' % (dn, nn_type, pooling_type, use_hie)


classifier_list = []

for cv_idx in range(5):
    print('Dev fold: %d' % cv_idx)
    save_fn = '%s/model-%d.ckpt' % (save_dn, cv_idx)
    # Define Model
    # Define Model
    if nn_type == 'conv':
        nn_kwargs = {'num_layers': 1, 'conv_size': 5}
    else:
        nn_kwargs = {'num_layers': 1, 'bidirectional': True}
    if pooling_type == 'attention':
        pooling_kwargs = {'hidden_dim': n_hidden, 'atten_dim': n_hidden}
    else:
        pooling_kwargs = {}    
    layer_info = {'nn_type': nn_type, 
                  'nn_kwargs': nn_kwargs, 
                  'dropout_p': 0.5, 
                  'pooling_type': pooling_type, 
                  'pooling_kwargs': pooling_kwargs}
    
    if use_hie:
        classifier = HieNNClassifier(corpus.dic.size, n_emb, n_hidden, corpus.n_type, pre_embedding=None, 
                                     word2sent_info=layer_info, sent2doc_info=layer_info, state_pass=False)
    else:
        classifier = FlatNNClassifier(corpus.dic.size, n_emb, n_hidden, corpus.n_type, pre_embedding=None, 
                                      word2doc_info=layer_info)
    classifier.load_state_dict(torch.load(save_fn))
    classifier_list.append(classifier)

vc = VotingClassifier(classifier_list)
vc.eval()
vc.to(device)

test_err = eval_batches(vc, test_batches)
print('Accuracy: %.4f' % (1 - test_err))

#for test_k in range(5):
#    with open(os.path.join(save_dn, 'test-fold-%d.pkl' % test_k), 'rb') as f:
#        test_x = pickle.load(f)
#        test_mask = pickle.load(f)
#        test_y = pickle.load(f)
#        
#    # load model
#    model_list = []
#    for valid_k in range(5):
#        model = load_model(os.path.join(save_dn, 'model-%d-%d' % (test_k, valid_k)), model_type)
#        model_list.append(model)
#    voting_model = VotingClassifier(model_list, voting=voting)
#    
#    # prediction
#    voting_res = []
#    keep_tail = False if model_type == 'cnn' else True
#    test_idx_batches = get_minibatches_idx(len(test_x), 32, keep_tail=keep_tail)
#    test_y = np.concatenate([test_y[idx_batch] for idx_batch in test_idx_batches])
#    
#    for batch_idx_seq in test_idx_batches:
#        voting_res.append(voting_model.predict(estimator_args=(test_x[batch_idx_seq], test_mask[batch_idx_seq])))
#    voting_res = np.concatenate(voting_res)
#    
#    # calc metrics
#    confus_matrix = np.array([[np.sum((test_y==1) & (voting_res==1)), np.sum((test_y==1) & (voting_res==0))],
#                              [np.sum((test_y==0) & (voting_res==1)), np.sum((test_y==0) & (voting_res==0))]])
#    accuracy = (confus_matrix[0, 0]+confus_matrix[1, 1]) / confus_matrix.sum()
#    precision = confus_matrix[0, 0] / confus_matrix[:, 0].sum()
#    recall = confus_matrix[0, 0] / confus_matrix[0, :].sum()
#    metrics_list.append([confus_matrix, accuracy, precision, recall])
#
#
#micro_accuracy = np.mean([metrics[1] for metrics in metrics_list])
#micro_precision = np.mean([metrics[2] for metrics in metrics_list])
#micro_recall = np.mean([metrics[3] for metrics in metrics_list])
#print('Accuracy: %.4f, Precision: %.4f, Recall: %.4f' % (micro_accuracy, micro_precision, micro_recall))

#======================= Prediction with ensembled model =====================#
# NOTE: sentence with length shorter than conv_size would cause NaN. 
#sent_list = ["i like it very much.", 
#             "i do not like it.", 
#             "it is so interesting.",
#             "it isn't interesting."]
#
#with open(os.path.join(save_dn, 'test-fold-0.pkl'), 'rb') as f:
#    test_x = pickle.load(f)
#    test_mask = pickle.load(f)
#    test_y = pickle.load(f)
#
#model_list = []
#for test_k in range(5):
#    model = load_model(os.path.join(save_dn, 'model-%d' % test_k), model_type)
#    model_list.append(model)
#voting_model = VotingClassifier(model_list, voting='hard')
#
#for sent in sent_list:
#    res = voting_model.predict_sent(sent, with_prob=True)
#    print('%s -> %s' % (sent, res))
#
#print(voting_model.predict(estimator_args=(test_x[:32], test_mask[:32]), with_prob=False))
#print(voting_model.predict(estimator_args=(test_x[:32], test_mask[:32]), with_prob=True))
#
##========================= Prediction with single model ======================#
#model = load_model(os.path.join(save_dn, 'model-0'), model_type)
#
#for sent in sent_list:
#    res = model.predict_sent(sent, with_prob=True)
#    print('%s -> %s' % (sent, res))