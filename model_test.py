import math
import numpy as np
import pickle

with open('result.pkl', 'rb') as file:
    dict = pickle.load(file)

f = open('load.para', 'rb')
para_load = pickle.load(f)
user_num = para_load['user_num']  # total number of users
item_num = para_load['item_num']  # total number of items

train_matrix = para_load['train_matrix']
train_matrix.data = np.array(train_matrix.data, dtype=np.int8)
train_matrix = train_matrix.toarray()  # the 0-1 matrix of training set
test_matrix = para_load['test_matrix']
test_matrix.data = np.array(test_matrix.data, dtype=np.int8)
test_matrix = test_matrix.toarray()  # the 0-1 matrix of testing set


item_ids = np.array(list(range(item_num)))
item_matrix = np.load('tgt_mtx.npy')

R = 0
NDCG = 0
eva_size = 10


def IDCG(num):
    if num == 0:
        return 1
    idcg = 0
    for i in list(range(num)):
        idcg += 1/math.log(i+2, 2)
    return idcg


def descend_sort(array):
    return -np.sort(-array)


null_user = 0

for user_id, row in enumerate(test_matrix):
    if row.sum() == 0:
        null_user += 1
        continue
    test_ids = item_matrix[np.where(row == 1), :]
    beams = dict[user_id][:, 1:]  # generated item id
    hit_num = 0
    dcg = 0
    for i, beam in enumerate(beams):
        if (beam == test_ids).all(-1).any():
            hit_num = hit_num + 1
            dcg = dcg + 1 / math.log(i + 2, 2)
    R += hit_num / np.sum(row)
    NDCG += dcg / IDCG(np.sum(descend_sort(row)[0:eva_size]))

R = R/(user_num - null_user)
NDCG = NDCG/(user_num - null_user)
print('R@%d: %.4f; NDCG@%d: %.4f' % (eva_size, R, eva_size, NDCG))
