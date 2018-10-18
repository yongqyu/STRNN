# STRNN
[Predicting the Next Location: A Recurrent Model with Spatial and Temporal Contexts](https://pdfs.semanticscholar.org/5bdf/0970034d0bb8a218c06ba3f2ddf97d29103d.pdf)  
Qiang Liu, Shu Wu, Liang Wang, Tieniu Tan (AAAI-16)


(recall@1:  0.0017160630653176505)  
(recall@5:  0.003539380072217654)  
(recall@10:  0.004004147152407851)  
(recall@100:  0.00464767080190197)  
(recall@1000:  0.01394045534150613)  
(recall@10000:  0.07866900175131349)  

## requirements
1.python 3.5  
2.pytorch 0.4.0

## Usage

### 0. Preprocessing
If you use (`prepro_xxx_50.txt`), you do not need to preprocess.  
If you want to perform personally or modify the source,
[Gowalla](http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz) is required at `../dataset/`.
```bash
$ python preprocess.py
```

### 1. Training
```bash
$ python train_torch.py
```
