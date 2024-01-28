import numpy as np
import pickle

f_para = open('load.para', 'rb')
para = pickle.load(f_para)
user_num = para['user_num']  # total number of users
item_num = para['item_num']  # total number of items
train_matrix = para['train_matrix']

train_matrix.data = np.array(train_matrix.data, dtype=np.int8)
train_matrix = train_matrix.toarray()  # the 0-1 matrix of testing set
u_len = np.sum(train_matrix, axis=1)

src_len = 32  # enc_input max sequence length
src_mtx = np.zeros((user_num, src_len), dtype=np.int)  # each row is a user token
index = np.arange(item_num)  # item ids: 0 ~ item_num

for u, row in enumerate(train_matrix):
    itr = index[np.where(row == 1)] + 1  # pad is 0
    if len(itr) >= src_len:
        u_src = np.random.choice(itr, size=src_len, replace=False)
        # u_src = np.sort(u_src)
    else:
        # itr = np.sort(itr)
        u_src = np.append(itr, np.zeros(src_len - len(itr), dtype=np.int))
    src_mtx[u] = u_src

np.save('src_mtx.npy', src_mtx)
print('finish!')