import pickle
import torch
import numpy as np
import torch.utils.data as D
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import time
import torch.utils.data as Data


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(128, 128)
        self.conv2 = SAGEConv(128, 64)
        self.USER_NUM = 36656
        self.ITEM_NUM = 76085
        self.fuse = nn.Linear(384, 128, bias=False)
        self.user = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(self.USER_NUM, 128)))

    def forward(self, feature, edge_index):
        tensor_list = [self.user, self.fuse(feature)]
        x_0 = torch.cat(tensor_list, dim=0)

        x_1 = self.conv1(x_0, edge_index)
        x_1 = F.leaky_relu(x_1)
        x_1 = F.dropout(x_1, p=0.5, training=self.training)

        x_2 = self.conv2(x_1, edge_index)
        x_2 = F.dropout(x_2, p=0.2, training=self.training)

        return x_2


f_para = open('load.para', 'rb')
para_load = pickle.load(f_para)
user_num = para_load['user_num']  # total number of users
item_num = para_load['item_num']  # total number of items
train_ui = para_load['train_ui']
print('total number of users is ', user_num)
print('total number of items is ', item_num)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

v_f_lookup = np.load('v_f.npy')  # size=(item_num, 128)
a_f_lookup = np.load('a_f.npy')
t_f_lookup = np.load('t_f.npy')
f_lookup = torch.from_numpy(np.concatenate((v_f_lookup, a_f_lookup, t_f_lookup), axis=1)).float().to(device)

batch_size = 3000
step_threshold = 500
epoch_max = 60
data_block = 6

train_i = torch.empty(0).long()
train_j = torch.empty(0).long()
train_m = torch.empty(0).long()
for b_i in list(range(data_block)):
    triple_para = pickle.load(open('triple_' + str(b_i) + '.para', 'rb'))
    train_i = torch.cat((train_i, torch.tensor(triple_para['train_i'])))  # 1-D tensor of user node ID
    train_j = torch.cat((train_j, torch.tensor(triple_para['train_j'])))  # 1-D tensor of pos item node ID
    train_m = torch.cat((train_m, torch.tensor(triple_para['train_m'])))  # 1-D tensor of neg item node ID

train_dataset = D.TensorDataset(train_i, train_j, train_m)
train_loader = D.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

gcn = GCN().to(device)
optimizer = torch.optim.Adam([{'params': gcn.parameters()}], lr=1e-3, weight_decay=1e-6)

train_ui = train_ui + [0, user_num]
edge_index = np.concatenate((train_ui, train_ui[:, [1, 0]]), axis=0)
edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_index = edge_index.t().contiguous().to(device)

gcn.train()

for epoch in range(epoch_max):
    running_loss = 0.0
    for step, (batch_i, batch_j, batch_m) in enumerate(train_loader):

        out = gcn(f_lookup, edge_index)

        embedding_i = out[batch_i.numpy(), :]
        embedding_j = out[batch_j.numpy() + user_num, :]
        embedding_m = out[batch_m.numpy() + user_num, :]

        predict_ij = torch.sum(torch.mul(embedding_i, embedding_j), dim=1)  # 1-D
        predict_im = torch.sum(torch.mul(embedding_i, embedding_m), dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(predict_ij - predict_im) + 1e-10))
        loss = bpr_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % step_threshold == (step_threshold - 1):
            print('[%d, %d] loss: %.5f' % (epoch + 1, step + 1, running_loss / step_threshold))
            running_loss = 0.0

gcn.eval()

with torch.no_grad():
    out = out = gcn(f_lookup, edge_index)
    out = out.cpu().numpy()
    np.save('representation.npy', out)


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx]


class RQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_embedding_1, n_embedding_2, n_embedding_3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.vq_embedding_1 = nn.Embedding(n_embedding_1, output_dim)
        self.vq_embedding_1.weight.data.uniform_(-1.0 / n_embedding_1, 1.0 / n_embedding_1)
        self.vq_embedding_2 = nn.Embedding(n_embedding_2, output_dim)
        self.vq_embedding_2.weight.data.uniform_(-1.0 / n_embedding_2, 1.0 / n_embedding_2)
        self.vq_embedding_3 = nn.Embedding(n_embedding_3, output_dim)
        self.vq_embedding_3.weight.data.uniform_(-1.0 / n_embedding_3, 1.0 / n_embedding_3)

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        ze_1 = self.encoder(x)
        # ze_1 = x
        # ze: [N, C]
        # embedding: [K, C]

        # block 1
        N, C = ze_1.shape
        embedding_1 = self.vq_embedding_1.weight.data
        K_1, _ = embedding_1.shape
        embedding_broadcast_1 = embedding_1.reshape(1, K_1, C)
        ze_broadcast_1 = ze_1.reshape(N, 1, C)
        distance_1 = torch.sum((embedding_broadcast_1 - ze_broadcast_1)**2, 2)
        # N
        nearest_1 = torch.argmin(distance_1, 1)
        # zq: [N, C]
        zq_1 = self.vq_embedding_1(nearest_1)

        # block 2
        ze_2 = ze_1 - zq_1
        embedding_2 = self.vq_embedding_2.weight.data
        K_2, _ = embedding_2.shape
        embedding_broadcast_2 = embedding_2.reshape(1, K_2, C)
        ze_broadcast_2 = ze_2.reshape(N, 1, C)
        distance_2 = torch.sum((embedding_broadcast_2 - ze_broadcast_2) ** 2, 2)
        # N
        nearest_2 = torch.argmin(distance_2, 1)
        # zq: [N, C]
        zq_2 = self.vq_embedding_2(nearest_2)

        # block 3
        ze_3 = ze_2 - zq_2
        embedding_3 = self.vq_embedding_3.weight.data
        K_3, _ = embedding_3.shape
        embedding_broadcast_3 = embedding_3.reshape(1, K_3, C)
        ze_broadcast_3 = ze_3.reshape(N, 1, C)
        distance_3 = torch.sum((embedding_broadcast_3 - ze_broadcast_3) ** 2, 2)
        # N
        nearest_3 = torch.argmin(distance_3, 1)
        # zq: [N, C]
        zq_3 = self.vq_embedding_3(nearest_3)

        decoder_input = ze_1 + ((zq_1 + zq_2 + zq_3) - ze_1).detach()
        x_hat = self.decoder(decoder_input)
        return x_hat, ze_1, ze_2, ze_3, zq_1, zq_2, zq_3, nearest_1, nearest_2, nearest_3


batch_size = 1024
lr = 1e-3
n_epochs = 60
l_w_embedding = 1
l_w_commitment = 0.25
content = np.load('representation.npy')[user_num: user_num + item_num, :]
loader = Data.DataLoader(MyDataSet(torch.Tensor(content)), batch_size=batch_size, shuffle=True)
model = RQVAE(64, 16, 4, 128, 128, 128).cuda()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)
mse_loss = nn.MSELoss()
tic = time.time()
for e in range(n_epochs):
    total_loss = 0
    for x in loader:
        current_batch_size = x.shape[0]
        x = x.cuda()
        x_hat, ze_1, ze_2, ze_3, zq_1, zq_2, zq_3, _, _, _ = model(x)
        l_reconstruct = mse_loss(x, x_hat)
        l_embedding = mse_loss(ze_1.detach(), zq_1) + mse_loss(ze_2.detach(), zq_2) + mse_loss(ze_3.detach(), zq_3)
        l_commitment = mse_loss(ze_1, zq_1.detach()) + mse_loss(ze_2, zq_2.detach()) + mse_loss(ze_3, zq_3.detach())
        loss = l_reconstruct + l_w_embedding * l_embedding + l_w_commitment * l_commitment
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * current_batch_size
    total_loss /= len(loader.dataset)
    toc = time.time()
    print(f'epoch {e} loss: {total_loss:.5f} elapsed {(toc - tic):.2f}s')


model.eval()
with torch.no_grad():
    id_1 = np.empty(shape=(0))
    id_2 = np.empty(shape=(0))
    id_3 = np.empty(shape=(0))
    res = np.empty(shape=(0, 64))

    for batch in np.array_split(np.array(list(range(item_num))), indices_or_sections=4):
        x = torch.Tensor(content[batch, :]).cuda()
        _, _, _, _, _, _, _, nearest_neighbor_1, nearest_neighbor_2, nearest_neighbor_3 = model(x)
        id_1 = np.append(id_1, nearest_neighbor_1.cpu().numpy())
        id_2 = np.append(id_2, nearest_neighbor_2.cpu().numpy())
        id_3 = np.append(id_3, nearest_neighbor_3.cpu().numpy())
    id_1 = id_1.reshape(-1, 1)
    id_2 = id_2.reshape(-1, 1)
    id_3 = id_3.reshape(-1, 1)
    r_id = np.concatenate((id_1, id_2, id_3), axis=1) + 1
    r_id = r_id.astype(np.int16)


result, inverse_indices = np.unique(r_id, axis=0, return_inverse=True)
col = np.zeros(r_id.shape[0], dtype=np.int)

train_matrix = para_load['train_matrix']
train_matrix.data = np.array(train_matrix.data, dtype=np.int8)
train_matrix = train_matrix.toarray()  # the 0-1 matrix of training set
popularity = np.sum(train_matrix, axis=0)

for i in range(result.shape[0]):
    loc = np.where(inverse_indices == i)[0]  #
    pop = popularity[loc]
    loc = loc[(-1 * pop).argsort()]

    for j, position in enumerate(loc):
        col[position] = j + 1

col = col.reshape((-1, 1))
tgt_mtx = np.concatenate((r_id, col), axis=1)
np.save('tgt_mtx.npy', tgt_mtx)
print('finish!')