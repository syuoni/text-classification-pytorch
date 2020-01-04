# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold
from gensim import corpora
import pickle
import jieba
import re
from collections import Counter


def tokenize_zh(sent, lower=True):
    if lower is True:
        sent = sent.lower()
    return [w for w in jieba.cut(sent) if not re.fullmatch('\s+', w)]


def cut_doc_set(doc_set, pat2word=None):
    '''
    doc_set: a list of str (sent)
    return: a list of list of str (word)
    '''
    new_doc_set = []
    for doc in doc_set:
        cutted = tokenize_zh(doc)
        if pat2word is None:
            new_doc_set.append(cutted)
        else:
            cutted_mapped = []
            for word in cutted:
                for pat in pat2word:
                    if re.fullmatch(pat, word):
                        word = pat2word[pat]
                cutted_mapped.append(word)
            new_doc_set.append(cutted_mapped)
    return new_doc_set


class Dictionary(object):
    def __init__(self, doc_set, pat2word=None, freq_crit=5):
        '''
        doc_set: a list of list of str (word)
        
        pat2word: a dict, like {'\d+\.?\d*': '#NUMBER'}
        The dictionary will map words matching the pattern described by the key to its value. 
        '''
        # Replace low-frequncy words with <UNK>
        freq = Counter([w for doc in doc_set for w in doc])        
        doc_set = [[w if freq[w] >= freq_crit else '<UNK>' for w in doc] for doc in doc_set]
        
        # Build dictionay
        # <UNK> Unknow: 0
        # <EOS> end-of-sentence: 1
        self.gensim_dic = corpora.Dictionary([['<UNK>'], ['<EOS>']])
        self.gensim_dic.add_documents(doc_set)
        
        self._word2idx = self.gensim_dic.token2id
        self._idx2word = {idx: w for w, idx in self._word2idx.items()}
        self._pat2word = pat2word
        self.size = len(self._idx2word)
        
    def word2idx(self, word, use_pat=False):
        '''use_pat=True for really unseen document. 
        '''
        if use_pat and (self._pat2word is not None):
            for pat in self._pat2word:
                if re.fullmatch(pat, word):
                    word = self._pat2word[pat]      
        # 0 for <UNK>
        return self._word2idx.get(word, 0)
        
    def word_seq2idx_seq(self, word_seq):
        return [self.word2idx(w) for w in word_seq]
        
    def idx_seq2word_seq(self, idx_seq):
        return [self._idx2word.get(idx, '') for idx in idx_seq]
    
    def word_seq2bow(self, word_seq):
        return list(Counter(self.word_seq2idx_seq(word_seq)).items())
        
    def sent2idx_seq(self, sent):
        return self.word_seq2idx_seq(tokenize_zh(sent, lower=True))
    
    def __repr__(self):
        dic_head = '; '.join(['%d: %s' % (idx, self._idx2word[idx]) for idx in range(5)])
        return 'Dictionary: %d words. \n\t%s... ' % (self.size, dic_head)
    
    
class Corpus(object):
    def __init__(self, df, pat2word=None):
        self.df = df
        self.pat2word = pat2word
        
        # df MUST contain columns: ['rating', 'w_seq']
        assert ('w_seq' in self.df.columns) and ('rating' in self.df.columns)
        
        targets = self.df['rating'].unique()
        assert (targets.min() == 0) and (targets.max() == len(targets)-1)
        
        if 'w_seq_len' not in self.df.columns:
            self.df['w_seq_len'] = self.df['w_seq'].str.len()
            
        self.size = self.df.shape[0]
        self.n_target = len(targets)
        
        
        if 'part_default' not in self.df.columns:
            # NO test part by default
            self.df['part_default'] = 'train'
            self.df.loc[self.df.sample(frac=0.2).index, 'part_default'] = 'dev'
            
        self.part_cols = []
        self.dics = {}
        self.split_sizes = {}
        self._add_part('part_default')
        self.set_current_part(part='part_default')
        
        
    def set_current_part(self, part='part_default'):
        if isinstance(part, (str, int)):
            if isinstance(part, str):
                part_col = part
            else:
                part_col = 'part_cv_%d' % part
            assert part_col in self.part_cols
            self.current_part_col = part_col
        else:
            raise Exception('Invalid part input!', part)
        
        self.current_dic = self.dics[self.current_part_col]
        self.current_split_sizes = self.split_sizes[self.current_part_col]
        
        
    def _add_part(self, part_col):
        self.part_cols.append(part_col)
        doc_set = self.df.loc[self.df[part_col] == 'train', 'w_seq'].tolist()
        self.dics[part_col] = Dictionary(doc_set, pat2word=self.pat2word)
        self.split_sizes[part_col] = ((self.df[part_col] == 'train').sum(),
                                      (self.df[part_col] == 'dev').sum(),
                                      (self.df[part_col] == 'test').sum())
        
    def _delete_part(self, part_col):
        self.part_cols.remove(part_col)
        del self.dics[part_col]
        del self.split_sizes[part_col]
        
        
    def create_cv(self, n_splits=5):
        '''Re-split train and dev sets as cross validation. 
        The test set, if existing, remains as the default partition. 
        '''
        train_dev_idxs = self.df.index[self.df['part_default'].isin(['train', 'dev'])].values
        cv = KFold(n_splits=n_splits, shuffle=True)
        for cv_idx, (train, dev) in enumerate(cv.split(train_dev_idxs)):
            cv_part_col = 'part_cv_%d' % cv_idx
            self.df[cv_part_col] = self.df['part_default']
            self.df.loc[train_dev_idxs[train], cv_part_col] = 'train'
            self.df.loc[train_dev_idxs[dev], cv_part_col] = 'dev'
            
            self._add_part(cv_part_col)
            
    
    def get_fullset(self, from_parts=None, return_type='seq'):
        '''
        return_type = 'seq' / 'sequence': word-index sequence
                    = 'bow': bag-of-words
        '''
        if from_parts is None:
            from_parts = ['train']
            
        this_df = self.df.loc[self.df[self.current_part_col].isin(from_parts)]
        
        if return_type.startswith('seq'):
            this_x = [self.current_dic.word_seq2idx_seq(w_seq) for w_seq in this_df['w_seq'].tolist()]
        elif return_type == 'bow':
            this_x = [self.current_dic.word_seq2bow(w_seq) for w_seq in this_df['w_seq'].tolist()]
        else:
            raise Exception('Invalid return type!', return_type)
            
        this_y = this_df['rating'].tolist()
        this_lens = this_df['w_seq_len'].tolist()
        return (this_x, this_y, this_lens)
    
        
    def iter_as_batches(self, batch_size=64, order='shuffle', from_parts=None, 
                        input_df=None, return_type='seq'):
        '''
        return_type = 'seq' / 'sequence': word-index sequence
                    = 'bow': bag-of-words
        input_df must be a DataFrame with columns 'w_seq' and 'w_seq_len'
        '''
        if input_df is not None:
            this_df = input_df
        else:
            if from_parts is None:
                from_parts = ['train']
            this_df = self.df.loc[self.df[self.current_part_col].isin(from_parts)]
            
        if order == 'shuffle':
            this_df = this_df.sample(frac=1)
        elif order == 'descending':
            # Most efficient computation
            this_df = this_df.sort_values(by='w_seq_len', ascending=False)
            
            
        n_sample = this_df.shape[0]
        n_batch = n_sample // batch_size
        if n_sample % batch_size > 0:
            n_batch += 1
        
        # Note (20190430): Originally written as range(1, n_batch), why?
        for batch_idx in range(n_batch):
            batch = this_df.iloc[batch_idx*batch_size:(batch_idx+1)*batch_size]
            
            if return_type.startswith('seq'):
                batch_x = [self.current_dic.word_seq2idx_seq(w_seq) for w_seq in batch['w_seq'].tolist()]
            elif return_type == 'bow':
                batch_x = [self.current_dic.word_seq2bow(w_seq) for w_seq in batch['w_seq'].tolist()]
            else:
                raise Exception('Invalid return type!', return_type)
            
            if 'rating' in batch.columns:
                batch_y = batch['rating'].tolist()
            else:
                batch_y = None
            batch_lens = batch['w_seq_len'].tolist()
            
#            maxlen = max(batch_lens)
#            batch_x = [x + [0] * (maxlen-len(x)) for x in batch_x]
            yield (batch_x, batch_y, batch_lens)
            
            
    def __repr__(self):
        outs = []
        outs.append('Corpus: %d documents, %d targets. ' % (self.size, self.n_target))
        for part_col in self.part_cols:
            this_out = 'Part: %s.' % part_col
            this_out = this_out + ('(current)' if part_col == self.current_part_col else '')
            outs.append(this_out)
            outs.append('\tTrain: %d, Dev: %d, Test: %d' % self.split_sizes[part_col])
            outs.append('\t%s' % self.dics[part_col])
        return '\n'.join(outs)
    
    
    @staticmethod
    def load_from_file(fn):
        '''Load prepared data and construct a Corpus object. 
        Use pickle to save/load an existing Corpus object. 
        '''
        with open(fn, 'rb') as f:
            df = pickle.load(f)
            pat2word = pickle.load(f)
        return Corpus(df, pat2word=pat2word)
        
        
if __name__ == '__main__':
    import torch
    from utils import sort_in_descending, _test_sort_in_descending, cast_to_tensor
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    dataset = 'imdb-2-200'
    corpus = Corpus.load_from_file('dataset/%s-prep.pkl' % dataset)
    for batch in corpus.iter_as_batches(batch_size=100, shuffle=True, from_parts=['test']):
        
        batch_x, batch_y, batch_lens = cast_to_tensor(batch, device)
        sorted_batch_lens, order, revert_order = sort_in_descending(batch_lens)
        _test_sort_in_descending(batch_lens)
        print(batch_x.size(0), batch_lens.max().item())
        break
        
#    corpus.create_cv()
#    corpus.set_current_part(1)
#    print(corpus)
    
#    doc_set = ['那边有5679999.5个问题。', 'TTT也有问题吗0.6']
#    print(cut_doc_set(doc_set, pat2word=pat2word))
    
