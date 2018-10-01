# Text classification
This project implements neural networks for text classification, based on PyTorch (0.4.0).  

# Data 
The original datasets are released by Tang et al. (2015). 

# Results  
## Test accuracies
|Model| Pooling |Hierarchical|IMDB(2)|IMDB(10)|Yelp 2013|Yelp 2014|
|:---:|:-------:|:----------:|:-----:|:------:|:-------:|:-------:|
|GRNN |Mean     |False       | 0.9121|  0.4691| | |
|GRNN |Max      |False       | 0.9246|  0.4781| | |
|GRNN |Attention|False       | 0.9259|  0.4855| | |
|GRNN |Mean     |True        | 0.9121|  0.4691| | |
|GRNN |Max      |True        | 0.9246|  0.4781| | |
|GRNN |Attention|True        | 0.9259|  0.4855| | |
|LSTM |Mean     |False       | 0.9068|  0.3855| | |
|LSTM |Max      |False       | 0.9234|  0.4836| | |
|LSTM |Attention|False       | 0.9233|  0.4813| | |
|CNN  |Mean     |False       | 0.9077|  0.4011| | |
|CNN  |Max      |False       | 0.9204|  0.4884| | |
|CNN  |Attention|False       | 0.9198|  0.4795| | |


# References
Tang, D., Qin, B., & Liu, T. (2015). Learning semantic representations of users and products for document level sentiment classification. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (Vol. 1, pp. 1014-1023).  

Chen, H., Sun, M., Tu, C., Lin, Y., & Liu, Z. (2016). Neural sentiment classification with user and product attention. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (pp. 1650-1659).  

Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 1480-1489).  
