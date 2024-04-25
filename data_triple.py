import numpy as np
import pickle
import gc


f_para = open('load.para', 'rb')
para = pickle.load(f_para)
user_num = para['user_num']  # total number of users
item_num = para['item_num']  # total number of items
train_matrix = para['train_matrix']
train_ui = para['train_ui']

print('train triple started...')
ratio = 5  # the ratio of positive and negative
item_ids = np.array(list(range(item_num)))
train_triple = np.empty(shape=[0, 3], dtype=int)
mtx = np.array(train_matrix.todense())
para_index = 0
for ct, inter in enumerate(train_ui):
    user_id = inter[0]  # user_id is an 1-D numpy array
    bool_index = ~np.array(mtx[user_id, :], dtype=bool)
    can_item_ids = item_ids[bool_index]  # the id list of 0-value items
    a1 = np.random.choice(can_item_ids, size=ratio, replace=False)  # a1 is an 1-D numpy array
    inter = np.expand_dims(inter, axis=0)
    inter = np.repeat(inter, repeats=ratio, axis=0)  # size = ratio * 2
    a1 = np.expand_dims(a1, axis=1)
    triple = np.append(inter, a1, axis=1)  # size = ratio * 3
    train_triple = np.append(train_triple, triple, axis=0)
    if ct % 10000 == 9999:
        print('=====[%d] ten thousand completed=====' % ((ct + 1)/10000))
    if ct % 1e5 == (1e5 - 1) and ct < len(train_ui)-1:
        train_i = train_triple[:, 0]
        train_j = train_triple[:, 1]
        train_m = train_triple[:, 2]
        para = {}
        para['train_i'] = train_i
        para['train_j'] = train_j
        para['train_m'] = train_m
        pickle.dump(para, open('triple_'+str(para_index)+'.para', 'wb'))
        print('para'+str(para_index)+'saved...')
        train_triple = np.empty(shape=[0, 3], dtype=int)
        para_index += 1
        del para
        gc.collect()

train_i = train_triple[:, 0]
train_j = train_triple[:, 1]
train_m = train_triple[:, 2]
para = {}
para['train_i'] = train_i
para['train_j'] = train_j
para['train_m'] = train_m
pickle.dump(para, open('triple_'+str(para_index)+'.para', 'wb'))
print('data_triple finished...')
