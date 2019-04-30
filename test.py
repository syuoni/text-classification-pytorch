# -*- coding: utf-8 -*-
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from corpus import Corpus
from utils import sort_in_descending, _test_sort_in_descending, cast_to_tensor
from models import HieLayer
from pooling_layers import construct_pooling_layer

class TestCorpus(unittest.TestCase):
    def test_corpus(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        dataset = 'imdb-2-200'
        corpus = Corpus.load_from_file('dataset/%s-prep.pkl' % dataset)
        for batch in corpus.iter_as_batches(batch_size=100, shuffle=True, from_parts=['test']):
            batch_x, batch_y, batch_lens = cast_to_tensor(batch, device)
            sorted_batch_lens, order, revert_order = sort_in_descending(batch_lens)
            _test_sort_in_descending(batch_lens)
            
            self.assertEqual(batch_x.size(0), 100)
            self.assertEqual(batch_lens.max().item(), 200)
            break
        
#        df = pd.read_stata(r'example-data\womenwk.dta')
#        ols_md = OrdinaryLeastSquare(df, 'work', ['age', 'married', 'children', 'education'])
#        ols_md.fit(robust=True, show_res=False)
#        
#        self.assertEqual(ols_md.res_stats['method'], 'OLS')
#        self.assertTrue(ols_md.res_stats['robust'])        
#        self.assertAlmostEqual(ols_md.res_stats['adj-R-sq'], 0.20102407308444759)
#        self.assertAlmostEqual(ols_md.res_stats['F-stats'], 192.57650255037254)
#        
#        self.assertTrue(np.allclose(ols_md.res_table['coef'].values, 
#                                    np.array([0.01025522, 0.11111163, 0.11530842, 0.01860109, -0.2073227])))
#        self.assertTrue(np.allclose(ols_md.res_table['est.std.err'].values, 
#                                    np.array([0.00122355, 0.02267192, 0.0056978 , 0.00330056, 0.05345809])))
#        self.assertTrue(np.allclose(ols_md.res_table['t-statistic'].values, 
#                                    np.array([8.381508  , 4.90084717,20.23737107, 5.63573275, -3.87822915])))
#        self.assertTrue(np.allclose(ols_md.predict(df).values[:5],
#                                    np.array([0.31541467, 0.45898772, 0.37694598, 0.46924294,  0.6050618 ])))





if __name__ == '__main__':
#    unittest.main(verbosity=2)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    dataset = 'imdb'
    corpus = Corpus.load_from_file('dataset/%s-prep.pkl' % dataset)
    for batch in corpus.iter_as_batches(batch_size=100, shuffle=True, from_parts=['test']):
        batch_x, batch_y, batch_lens = cast_to_tensor(batch, device)
        sorted_batch_lens, order, revert_order = sort_in_descending(batch_lens)
        _test_sort_in_descending(batch_lens)
        
        break
    
    emb_layer = nn.Embedding(corpus.current_dic.size, 128)
    pooling_layer = construct_pooling_layer('attention',  pooling_kwargs={'hidden_dim': 128})
    hie_layer = HieLayer(pooling_layer=pooling_layer)
    
    emb_layer.to(device)
#    pooling_layer.to(device)
    hie_layer.to(device)
    
    embed = emb_layer(batch_x)
    doc_list1, doc_lens1 = hie_layer.forward(embed, batch_x, save_memory=True)
    doc_list2, doc_lens2 = hie_layer.forward(embed, batch_x, save_memory=False)
    
    import numpy as np
    np.isclose(doc_list1.detach().cpu().numpy(), doc_list2.detach().cpu().numpy(), atol=1e-6).all()
    