# Text classification
This project implements neural networks for text classification, based on PyTorch (0.4.0).  

## Test accuracy 
### Non-hierarchical structure
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

