# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle, gzip

maxlen = 200

# dic info
with gzip.open('dataset/theano/imdb.dict.pkl.gz', 'rb') as f:
    word2idx = pickle.load(f, encoding='bytes')
word2idx = {w.decode(): idx for w, idx in word2idx.items()}
word2idx['<UNK>'] = 0
word2idx['<EOS>'] = 1

# corpus info
with open('dataset/theano/imdb.pkl', 'rb') as f:
    train_x, train_y = pickle.load(f, encoding='bytes')
    test_x, test_y = pickle.load(f, encoding='bytes')

train_df = pd.DataFrame({'wid_seq': train_x, 
                         'rating': train_y, 
                         'part': 'train'})
test_df = pd.DataFrame({'wid_seq': test_x, 
                        'rating': test_y, 
                        'part': 'test'})

np.random.seed(1234)
rand = np.random.rand(test_df.shape[0])
test_df.loc[rand > np.median(rand), 'part'] = 'dev'
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

df['wid_seq_len'] = df['wid_seq'].str.len()
#df['wid_seq_len'].hist(bins=50)
print(df['wid_seq_len'].quantile(0.9), maxlen)
df = df.loc[df['wid_seq_len'] <= maxlen].reset_index(drop=True)

# Reset train-dev-test split
if maxlen <= 200:
    rand = np.random.rand(df.shape[0])
    df.loc[rand > np.percentile(rand, 85), 'part'] = 'test'
    df.loc[(rand > np.percentile(rand, 70)) & (rand <= np.percentile(rand, 85)), 'part'] = 'dev'
    df.loc[rand <= np.percentile(rand, 70), 'part'] = 'train'

pat2word = None

# Dump corpus
with open('dataset/imdb-2-%d-prep.pkl' % maxlen, 'wb') as f:
    pickle.dump(word2idx, f)
    pickle.dump(pat2word, f)
    pickle.dump(df, f)

