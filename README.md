# MMGRec
This is our experiment codes for the paper:

MMGRec: Multimodal Generative Recommendation with Transformer Model

## Environment settings
* Python 3.7
* Pytorch 1.7.0+cu101
* PyTorch Geometric 1.7.2
* Numpy 1.19.5

## File specification
* model_train.py : the training process of MMGRec.
* model_test.py : the testing process of MMGRec.

## Usage
* Execution sequence

  The execution sequence of codes is as follows: model_train.py--->model_test.py
  
* Execution results

  During the execution of file model_train.py, the epoch and training loss will be printed as the training process:
  
  ```
  Epoch: 0001 loss = 4.690832
  Epoch: 0002 loss = 3.858034
  Epoch: 0003 loss = 3.622033
  Epoch: 0004 loss = 3.485817
  ...
  ```

  File model_test.py should be executed after the training process, and the performance of MetaMMF_GCN will be printed:
  
  ```
  R@10: 0.2804; NDCG@10: 0.1898
  ```
