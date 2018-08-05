# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold
import pickle
import jieba
import re


class Dictionary(object):
    def __init__(self, word2idx, pat2word=None):
        '''pat2word could be a dict, like {'\d+\.?\d*': '#NUMBER'}
        the dictionary will map words matching the pattern described by the key to its value.   
        '''
        self._word2idx = word2idx
        self._idx2word = {idx: w for w, idx in self._word2idx.items()}
        self._pat2word = pat2word
        self.size = len(self._idx2word)
        assert max(self._idx2word) == self.size - 1
        assert min(self._idx2word) == 0
    
    def word2idx(self, word):
        if self._pat2word is not None:
            for pat in self._pat2word:
                if re.fullmatch(pat, word):
                    return self._word2idx.get(self._pat2word[pat])
        # idx of 0 is #UNDEF by default
        return self._word2idx.get(word, 0)
    
    def word_seq2idx_seq(self, word_seq):
        return [self.word2idx(w) for w in word_seq]

    def idx_seq2word_seq(self, idx_seq):
        return [self._idx2word.get(idx, '') for idx in idx_seq]
    
    def sent2idx_seq(self, sent):
        return self.word_seq2idx_seq(Dictionary.tokenize(sent, lower=True))
    
    @staticmethod
    def tokenize(sent, lower=True):
        if lower is True:
            sent = sent.lower()
        return [w for w in jieba.cut(sent) if not re.fullmatch('\s+', w)]
    
    
class Corpus(object):
    '''df MUST contain columns: ['part', 'rating', 'wid_seq', 'wid_seq_len']
    '''
    def __init__(self, df, word2idx, pat2word=None):
        self.df = df
        self.size = df.shape[0]
        
        assert df['rating'].min() == 0
        self.n_type = df['rating'].max().item() + 1
        self.dic = Dictionary(word2idx, pat2word=pat2word)
        
        self.current_part_col = 'part'
        self.cv_part_cols = [x for x in df.columns if x.startswith('part_cv')]
        self._update_part_sizes()
    
    def create_cv(self, n_splits=5):
        '''Re-split train and dev sets
        '''
        train_dev_idxs = self.df.index[self.df['part'].isin(['train', 'dev'])].values
        cv = KFold(n_splits=n_splits, shuffle=True)
        for cv_idx, (train, dev) in enumerate(cv.split(train_dev_idxs)):
            self.df['part_cv_%d' % cv_idx] = self.df['part']
            self.df.loc[train_dev_idxs[train], 'part_cv_%d' % cv_idx] = 'train'
            self.df.loc[train_dev_idxs[dev], 'part_cv_%d' % cv_idx] = 'dev'
        self.cv_part_cols = ['part_cv_%d' % cv_idx for cv_idx in range(n_splits)]
    
    def set_part(self, part='part'):
        if part == 'part':
            self.current_part_col = 'part'
        elif isinstance(part, str):
            assert part in self.cv_part_cols
            self.current_part_col = part
        elif isinstance(part, int):
            part = 'part_cv_%d' % part
            assert part in self.cv_part_cols
            self.current_part_col = part
        else:
            raise Exception('Invalid part input!', part)
        self._update_part_sizes()
            
    def _update_part_sizes(self):
        self.train_size = (self.df[self.current_part_col] == 'train').sum()
        self.dev_size = (self.df[self.current_part_col] == 'dev').sum()
        self.test_size = (self.df[self.current_part_col] == 'test').sum()
    
    def iter_as_batches(self, batch_size=64, shuffle=True, from_parts=None):
        if from_parts is None:
            from_parts = ['train']
            
        this_df = self.df.loc[self.df[self.current_part_col].isin(from_parts)]
        if shuffle is True:
            this_df = this_df.sample(frac=1)
        else:
            # If NOT shuffle, use descending order by default
            this_df = this_df.sort_values(by='wid_seq_len', ascending=False)
            
        n_sample = this_df.shape[0]
        n_batch = n_sample // batch_size
        if n_sample % batch_size > 0:
            n_batch += 1
        
        for batch_idx in range(1, n_batch):
            batch = this_df.iloc[batch_idx*batch_size:(batch_idx+1)*batch_size]
            
            batch_x = batch['wid_seq'].tolist()
            batch_y = batch['rating'].tolist()
            batch_lens = batch['wid_seq_len'].tolist()
            
#            maxlen = max(batch_lens)
#            batch_x = [x + [0] * (maxlen-len(x)) for x in batch_x]
            yield (batch_x, batch_y, batch_lens)
        
    def save(self, corpus_fn):
        with open(corpus_fn, 'wb') as f:
            pickle.dump(self.dic._word2idx, f)
            pickle.dump(self.dic._pat2word, f)
            pickle.dump(self.df, f)
    
    @staticmethod
    def load_from_file(fn):
        with open(fn, 'rb') as f:
            word2idx = pickle.load(f)
            pat2word = pickle.load(f)
            df = pickle.load(f)
        return Corpus(df, word2idx, pat2word)
        
        
if __name__ == '__main__':
    import torch
    from utils import sort_in_descending, _test_sort_in_descending, cast_to_tensor
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    dataset = 'imdb'    
    corpus = Corpus.load_from_file('dataset/%s-prep.pkl' % dataset)
    for batch in corpus.iter_as_batches(batch_size=100, shuffle=True, from_parts=['test']):
        
        batch_x, batch_y, batch_lens = cast_to_tensor(batch, device)
        sorted_batch_lens, order, revert_order = sort_in_descending(batch_lens)
        _test_sort_in_descending(batch_lens)
        print(batch_x.size(0), batch_lens.max().item())
        
        break
        
#    corpus.create_cv()
    
#    fn = r'imdb\imdb-prepared.pkl'
#    corpus = Corpus.load_from_file(fn)
#    corpus.save(r'imdb\imdb-resaved.pkl')
#    with open(r'weibo-hp\raw-corpus-relev_vs_norel.pkl', 'rb') as f:
#        data_x = pickle.load(f)
#        data_y = pickle.load(f)
#        
#    pat2word = {'\d{1}(\.\d*)?': '#NUM1',
#                '\d{2}(\.\d*)?': '#NUM2',
#                '\d{3}(\.\d*)?': '#NUM3',
#                '\d{4}(\.\d*)?': '#NUM4',
#                '\d{5,}(\.\d*)?': '#NUM5'}
#    
#    dump_to_fn = r'weibo-hp\corpus-relev_vs_norel.pkl'
#    Corpus.build_corpus_with_dic(data_x, data_y, 60, 2, dump_to_fn=dump_to_fn, pat2word=pat2word)
#    corpus = Corpus.load_from_file(dump_to_fn)
    
    
