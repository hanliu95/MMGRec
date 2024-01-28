# MMGRec
This is our experiment codes for the paper:

MMGRec: Multimodal Generative Recommendation with Transformer Model

## Environment settings
* Python 3.7
* Pytorch 1.7.0+cu101
* PyTorch Geometric 1.7.2
* Numpy 1.19.5

## File specification
* src_input.py : obtains the historical interaction history of users.
* tgt_input.py : obtains the Rec-IDs of items.
* model_train.py : the training process of MMGRec.
* model_test.py : the testing process of MMGRec.

## Usage
* Execution sequence

  The execution sequence of codes is as follows: src_input.py--->tgt_input.py--->model_train.py--->model_test.py
  
* Execution results

  During the execution of file model_train.py, the epoch and training loss will be printed as the training process:
  
  ```
  Epoch: 0001 loss = 4.164487
  Epoch: 0002 loss = 3.460217
  Epoch: 0003 loss = 3.060792
  Epoch: 0004 loss = 2.914330
  ...
  ```

  File model_test.py should be executed after the training process, and the performance of MMGRec will be printed:
  
  ```
  R@10: 0.1269; NDCG@10: 0.0802
  ```
