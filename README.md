# text classification
This project implements neural networks for text classification, based on PyTorch (0.4.0).  

## test accuracy 
|Model| Pooling |IMDB(2)|IMDB(10)|Yelp 2013|Yelp 2014|
|:---:|:-------:|:-----:|:------:|:-------:|:-------:|
|GRNN |Mean     |  |0.4691| | |
|GRNN |Max      |  |0.4781| | |
|GRNN |Attention|  |0.4855| | |
|LSTM |Mean     |  |0.3655| | |
|LSTM |Max      |  |0.4836| | |
|LSTM |Attention|  |0.4813| | |
|CNN  |Mean     |  |0.4011| | |
|CNN  |Max      |  |0.4884| | |
|CNN  |Attention|  | | | | |

