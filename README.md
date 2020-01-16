# Text Classification
This project implements neural networks for text classification, based on PyTorch (0.4.0).  

# Data 
The original datasets are released by Tang et al. (2015). 

# Configuration
* Word Embedding Layer
    * Layer Number: 1
    * Dimension: 128
    * Pre-Training: Word2Vec / Wiki
* Hidden Layer
    * Bidirectional (GRNN / LSTM): True
    * Convolutional Size (CNN): 5
    * Dimension: 128 (64 for each direction for GRNN / LSTM)
* Mini-Batch Size: 32
* Optimization Algorithm
    * Adadelta: lr=1.0, rho=0.9
    * SGD: lr=0.001, momentum=0.9
* Performance Evaluation
    * Leave out the Testing Set
    * Five-Fold Cross Validation on the Training Set
        * Four Folds as Training Set
        * One Fold as Validation Set for Early-Stopping
    * Combine the Five Classifiers by Soft-Voting 
    * Predict on the Testing Set

# Results  
## Testing Accuracies
|Model| Pooling  |Hierarchical|IMDB(2)|IMDB(10)|Yelp 2013|Yelp 2014|
|:---:|:--------:|:----------:|:-----:|:------:|:-------:|:-------:|
|GRNN |Mean      |False|  0.9174  |  0.4992  |**0.6483**|  0.6448  |
|GRNN |Max       |False|**0.9212**|  0.5008  |**0.6483**|  0.6495  |
|GRNN |Attention |False|  0.9192  |  0.5038  |  0.6423  |**0.6501**|
|LSTM |Mean      |False|  0.9095  |  0.4789  |  0.6321  |  0.6409  |
|LSTM |Max       |False|  0.9172  |  0.5021  |**0.6483**|  0.6484  |
|LSTM |Attention |False|  0.9152  |**0.5042**|  0.6414  |  0.6483  |
|CNN  |Mean      |False|  0.9046  |  0.4403  |  0.6191  |  0.6216  |
|CNN  |Max       |False|  0.9163  |  0.4956  |  0.6333  |  0.6333  |
|CNN  |Attention |False|  0.9129  |  0.4922  |  0.6303  |  0.6310  |
|GRNN |Mean      |True |    |    |    |    |
|GRNN |Max       |True |    |    |    |    |
|GRNN |Attention |True |    |    |    |    |


# References
Tang, D., Qin, B., & Liu, T. (2015). Learning semantic representations of users and products for document level sentiment classification. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (Vol. 1, pp. 1014-1023).  

Chen, H., Sun, M., Tu, C., Lin, Y., & Liu, Z. (2016). Neural sentiment classification with user and product attention. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (pp. 1650-1659).  

Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 1480-1489).  
