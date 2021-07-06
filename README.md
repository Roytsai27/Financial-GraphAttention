# FinGAT: A Financial Graph Attention Networkto Recommend Top-K Profitable Stocks
This is our implementation for the [paper](https://arxiv.org/abs/2106.10159):
FinGAT: A Financial Graph Attention Networkto Recommend Top-K Profitable Stocks


## Requirements
* pytorch==1.0.0
* numpy==1.16.4
* pandas==0.25.3

## Model architecture
![](https://i.imgur.com/lkCA1Rt.png)



## How to train the model
1. Run clean_data.py
This script would run the preprocessing for raw data and dump a preprocessed file.
2. Run train.py
you can tune the hyper parameters by adding args after train.py
e.g. python3 train.py --epoch 10 --l2 1e-6 etc.
```
--epoch: number of epochs
--l2: l2 regularization
--dim: dimension for hidden layer
--alpha: The adaptive weight on MAE loss
--beta: The adaptive weight on classification loss
--gamma: The adaptive weight on ranking loss
--lr: learning rate
--device: The device name for training, if train with cpu please use:"cpu"
```

## Result
![](https://i.imgur.com/ANEXmfH.png)
![](https://i.imgur.com/e8KmLKU.png)
![](https://i.imgur.com/DGuClLM.png)


