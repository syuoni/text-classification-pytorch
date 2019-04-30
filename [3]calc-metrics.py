# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch

from corpus import Corpus
from models import construct_classifier
from models import FlatNNClassifier, HieNNClassifier
from training import eval_batches
from predictors import Predictor

n_hidden = 128
n_emb = 128
batch_size = 32

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_hie = False


#dataset = 'imdb'
dataset = 'yelp-2013'
#dataset = 'yelp-2014'
corpus = Corpus.load_from_file('dataset/%s-prep.pkl' % dataset)

#dn = 'model-res-%s' % dataset
#corpus_fn = '%s/corpus-%s-with-cv.pkl' % (dn, dataset)
#with open(corpus_fn, 'rb') as f:
#    corpus = pickle.load(f)

#TODO: Align the dictionary/corpus for each train-test split...
test_batches = list(corpus.iter_as_batches(batch_size=batch_size*5, order='descending', from_parts=['test']))

#=========================== Metrics Calculation =============================#
'''
GRU, attetion, no-hie
Adadelta, 1.0:     |0.6261|0.6279|0.6230(rho=0.95)|

SGD, 1.0000: 0.4177|0.4173|0.4189
SGD, 0.5000: 0.5708|0.5600|0.5351
SGD, 0.2000: 0.6186|0.6031|0.6097
SGD, 0.1000: 0.6193|0.6222|0.6251
SGD, 0.0500: 0.6347|0.6218|0.6199
SGD, 0.0200: 0.6256|0.6302|0.6304
SGD, 0.0100: 0.6218|0.6325|0.6287
SGD, 0.0050: 0.4223|0.6324|0.6304
SGD, 0.0020: 0.4177|0.6354|0.6315
SGD, 0.0010: 0.4177|0.6330|0.6276
SGD, 0.0005:       |0.6318|0.6265
SGD, 0.0002:       |0.6259|0.6252
SGD, 0.0001:       |0.6270|0.6261


GRU, attention, hie
Adadelta, 1.0: 0.6342
'''

'''
|Model| Pooling |Hierarchical|IMDB(2)|IMDB(10)|Yelp 2013|Yelp 2014|
|:---:|:-------:|:----------:|:-----:|:------:|:-------:|:-------:|
|GRNN |Mean     |False       | |  | | |
|GRNN |Max      |False       | |  | | |
|GRNN |Attention|False       | |  | | |
|GRNN |Mean     |True        | |  | | |
|GRNN |Max      |True        | |  | | |
|GRNN |Attention|True        | |  | | |
|LSTM |Mean     |False       | |  | | |
|LSTM |Max      |False       | |  | | |
|LSTM |Attention|False       | |  | | |
|CNN  |Mean     |False       | |  | | |
|CNN  |Max      |False       | |  | | |
|CNN  |Attention|False       | |  | | |
'''
#=============================================================================#



nn_type = 'gru'
#nn_type = 'lstm'
#nn_type = 'conv'
#pooling_type = 'mean'
#pooling_type = 'max'
pooling_type = 'attention'

optim_type = 'SGD'
optim_type = 'Adadelta'
lr = 1.0

save_fn = 'model-res/model-%s-%s-%s-%s-%.4f-v3.ckpt' % (nn_type, pooling_type, use_hie, optim_type, lr)
classifier = construct_classifier(corpus.current_dic.size, n_emb, n_hidden, corpus.n_target, 
                                  pre_embedding=None, use_hie=use_hie, 
                                  nn_type=nn_type, pooling_type=pooling_type)
classifier.load_state_dict(torch.load(save_fn))
classifier.to(device)

#classifier_list = []
#for cv_idx in range(5):
#    print('Dev fold: %d' % cv_idx)
#    corpus.set_current_part(cv_idx)
#    
#    save_fn = '%s/model-%s-%s-%s-%d.ckpt' % (dn, nn_type, pooling_type, use_hie, cv_idx)
#    
#    classifier = construct_classifier(corpus.current_dic.size, n_emb, n_hidden, corpus.n_target, 
#                                      pre_embedding=None, use_hie=use_hie, 
#                                      nn_type=nn_type, pooling_type=pooling_type)
#    classifier.load_state_dict(torch.load(save_fn))
#    classifier_list.append(classifier)
#    
#vc = VotingClassifier(classifier_list)
##vc.eval()
#vc.to(device)

test_err = eval_batches(classifier, test_batches)
print('Accuracy: %.4f' % (1 - test_err))

df = corpus.df[['w_seq', 'w_seq_len']]
all_batches = list(corpus.iter_as_batches(batch_size=batch_size*5, order='original', input_df=df))
#all_err = eval_batches(classifier, all_batches)

predictor = Predictor(classifier, corpus)
x = predictor.decision_func_on_w_seq_df(df)


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