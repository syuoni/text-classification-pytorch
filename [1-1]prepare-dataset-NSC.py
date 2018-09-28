# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import Counter
import pickle

dataset = 'imdb'
maxlen = 800
#dataset = 'yelp-2013-seg-20-20'
#dataset = 'yelp-2014-seg-20-20'
#maxlen = 500

data = []
for part in ['train', 'dev', 'test']:
    if dataset == 'imdb':
        fn = 'dataset/NSC/%s.%s.txt.ss' % (dataset, part)
    else:
        fn = 'dataset/NSC/%s.%s.ss' % (dataset, part)
    
    with open(fn, encoding='utf-8') as f:
        for line in f:
            this_data = [x.strip() for x in line.split('\t\t')]
            this_data.append(part)
            data.append(this_data)
            
            
df = pd.DataFrame(data, columns=['user', 'product', 'rating', 'text', 'part_default'])
df['rating'] = df['rating'].astype(np.int)
df['rating'] = df['rating'] - 1

# Split each document into a word sequence; label <EOS>
doc_set = []
for text in df['text']:
    doc = [w if w != '<sssss>' else '<EOS>' for w in text.split(' ')]
    if doc[-1] != '<EOS>':
        doc.append('<EOS>')
    doc_set.append(doc)

# Count word frequency 
freq = Counter([w for doc in doc_set for w in doc])
freq_df = pd.DataFrame([[w, f] for w, f in freq.items()], columns=['word', 'freq'])
freq_df = freq_df.sort_values('freq', ascending=False)
freq_df.to_excel('dataset/freq/word-freq-%s.xlsx' % dataset, 'freq', index=False)

# Map pattern to word
pat2word = None

# Make Dataframe
df['w_seq'] = doc_set
df['w_seq_len'] = df['w_seq'].str.len()
#df['w_seq_len'].hist(bins=50)

print(df['w_seq_len'].quantile(0.9), maxlen)
df = df.loc[df['w_seq_len'] <= maxlen].reset_index(drop=True)

# Dump corpus
with open('dataset/%s-prep.pkl' % dataset, 'wb') as f:
    pickle.dump(df, f)
    pickle.dump(pat2word, f)
    
    