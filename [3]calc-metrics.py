# -*- coding: utf-8 -*-
import os
import itertools
import pickle
import numpy as np
import pandas as pd
import torch

from corpus import Corpus
from models import construct_classifier
from models import FlatNNClassifier, HieNNClassifier
from training import eval_batches
from predictors import Predictor, CVVotingPredictor

n_hidden = 128
n_emb = 128
batch_size = 32

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_hie = False


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

voting_acc_df = None

for dataset in ['imdb-2-200', 'imdb', 'yelp-2013', 'yelp-2014']:
    #dataset = 'imdb'
    #dataset = 'yelp-2013'
    #dataset = 'yelp-2014'
    
    dn = 'model-res-%s' % dataset
    corpus_fn = '%s/corpus-%s-with-cv.pkl' % (dn, dataset)
    with open(corpus_fn, 'rb') as f:
        corpus = pickle.load(f)
    
    # This is average out-of-sample accuracies. 
    #ave_accs = []
    #for nn_type, pooling_type in itertools.product(['gru', 'lstm', 'conv'], ['mean', 'max', 'attention']):
    #    #nn_type, pooling_type = 'gru', 'attention'
    #    print(nn_type, pooling_type)
    #
    #    cv_accs = []
    #    for cv_idx in range(5):
    #        # Align the dictionary/corpus for each train-test split
    #        print('Dev fold: %d' % cv_idx)
    #        corpus.set_current_part(cv_idx)
    #        
    #        save_fn = '%s/model-%s-%s-%s-%d.ckpt' % (dn, nn_type, pooling_type, use_hie, cv_idx)
    #        classifier = construct_classifier(corpus.current_dic.size, n_emb, n_hidden, corpus.n_target, 
    #                                          pre_embedding=None, use_hie=use_hie, 
    #                                          nn_type=nn_type, pooling_type=pooling_type)
    #        classifier.load_state_dict(torch.load(save_fn))
    #        classifier.to(device)
    #        
    #        # The test part is the same as that in the default partition. 
    #        test_batches = list(corpus.iter_as_batches(batch_size=batch_size*5, order='descending', from_parts=['test']))
    #        test_err = eval_batches(classifier, test_batches)
    #        cv_accs.append(1 - test_err)
    #        
    #    ave_accs.append([nn_type, pooling_type, sum(cv_accs) / 5])
    #
    #ave_acc_df = pd.DataFrame(ave_accs, columns=['nn_type', 'pooling_type', 'acc'])
    #ave_acc_df.to_excel('model-metrics/ave-acc.xlsx', 'acc', index=False)
    
    
    # This is out-of-sample accuracies by a voting predictor constructed by CV classifiers
    test_df = corpus.df.loc[corpus.df['part_default']=='test'].copy()
    voting_accs = []
    for nn_type, pooling_type in itertools.product(['gru', 'lstm', 'conv'], ['mean', 'max', 'attention']):
        #nn_type, pooling_type = 'gru', 'attention'
        print(dataset, nn_type, pooling_type)
        
        classifier = construct_classifier(corpus.current_dic.size, n_emb, n_hidden, corpus.n_target, 
                                          pre_embedding=None, use_hie=use_hie, 
                                          nn_type=nn_type, pooling_type=pooling_type)
        classifier.to(device)
        
        classifier_state_paths = ['%s/model-%s-%s-%s-%d.ckpt' % (dn, nn_type, pooling_type, use_hie, cv_idx) for cv_idx in range(5)]
        cvv_predictor = CVVotingPredictor(classifier, classifier_state_paths, corpus)
        
        y_pred = cvv_predictor.predict_on_w_seq_df(test_df)
        test_acc = (test_df['rating'].values == y_pred.cpu().numpy()).mean()
        voting_accs.append([nn_type, pooling_type, test_acc])
        
    this_voting_acc_df = pd.DataFrame(voting_accs, columns=['nn_type', 'pooling_type', dataset])
    if voting_acc_df is None:
        voting_acc_df = this_voting_acc_df
    else:
        voting_acc_df = pd.merge(voting_acc_df, this_voting_acc_df, on=['nn_type', 'pooling_type'], how='outer')


voting_acc_df.to_excel('model-metrics/voting-acc.xlsx', 'acc', index=False)


# Render for Markdown table
nn_type_dic = {'gru': 'GRNN', 
               'lstm': 'LSTM', 
               'conv': 'CNN'}
for i in voting_acc_df.index:
    this_row = [nn_type_dic[voting_acc_df.loc[i, 'nn_type']].ljust(5, " "), 
                voting_acc_df.loc[i, 'pooling_type'].capitalize().ljust(10, " "), 
                'False'.ljust(5, " ")]
    
    for dataset in ['imdb-2-200', 'imdb', 'yelp-2013', 'yelp-2014']:
        if dataset not in voting_acc_df.columns:
            this_acc = "    "
        elif voting_acc_df.loc[i, dataset] == voting_acc_df[dataset].max():
            this_acc = "**%.4f**" % voting_acc_df.loc[i, dataset]
        else:
            this_acc = "  %.4f  " % voting_acc_df.loc[i, dataset]
        this_row.append(this_acc)
        
    print("|".join(["", *this_row, ""]))
    


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