import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import pickle

src_mtx = np.load('src_mtx.npy')
tgt_mtx = np.load('tgt_mtx.npy')

P = 0
S = np.max(tgt_mtx) + 1
E = S + 1
tgt_vocab_size = E + 1

f_para = open('load.para', 'rb')
para = pickle.load(f_para)
train_ui = para['train_ui']
user_num = para['user_num']  # total number of users
item_num = para['item_num']  # total number of items

rep = np.load('representation.npy')
user_matrix = rep[: user_num, :]
item_matrix = rep[user_num: user_num + item_num, :]
src_emb_weight = torch.FloatTensor(np.concatenate((np.zeros((1, item_matrix.shape[1])), item_matrix), axis=0))
user_emb_weight = torch.FloatTensor(user_matrix)


# Transformer Parameters
d_model = 64  # Embedding Size
d_ff = 64  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 2  # number of Encoder of Decoder Layer
n_heads = 4  # number of heads in Multi-Head Attention
epoch_max = 1000
batch_size = 3000


def make_data():
    enc_inputs, dec_inputs, dec_outputs, u_inputs = [], [], [], []
    for pair in train_ui:
        enc_input = [src_mtx[pair[0]]]
        dec_input = [np.insert(tgt_mtx[pair[1]], 0, S)]
        dec_output = [np.append(tgt_mtx[pair[1]], E)]
        u_input = [[pair[0]]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
        u_inputs.extend(u_input)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs), torch.LongTensor(u_inputs)


enc_inputs, dec_inputs, dec_outputs, u_inputs = make_data()


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs, u_inputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
        self.u_inputs = u_inputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx], self.u_inputs[idx]


loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs, u_inputs), batch_size=batch_size, shuffle=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        output = self.dropout(output)
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.U_Q = nn.Linear(d_model, d_k, bias=False)
        self.U_K = nn.Linear(d_model, d_k, bias=False)
        self.U_V = nn.Linear(d_model, d_v, bias=False)
        self.sca_q = nn.Sequential(
            nn.Linear(d_model, d_model/2, bias=False),
            nn.ReLU(),
            nn.Linear(d_model/2, 1, bias=False)
        )
        self.sca_k = nn.Sequential(
            nn.Linear(d_model, d_model/2, bias=False),
            nn.ReLU(),
            nn.Linear(d_model/2, 1, bias=False)
        )
        self.sca_v = nn.Sequential(
            nn.Linear(d_model, d_model/2, bias=False),
            nn.ReLU(),
            nn.Linear(d_model/2, 1, bias=False)
        )
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.pos_ffn = PoswiseFeedForwardNet()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input, attn_mask, input_u):
        residual, batch_size = input, input.size(0)
        Q = self.W_Q(input).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        S_q = self.sca_q(input_u).repeat(1, input.size(1), d_k)
        S_k = self.sca_k(input_u).repeat(1, input.size(1), d_k)
        S_v = self.sca_v(input_u).repeat(1, input.size(1), d_k)
        u_Q = torch.mul(self.U_Q(input), S_q).view(batch_size, -1, 1, d_k).transpose(1, 2)
        u_K = torch.mul(self.U_K(input), S_k).view(batch_size, -1, 1, d_k).transpose(1, 2)
        u_V = torch.mul(self.U_V(input), S_v).view(batch_size, -1, 1, d_v).transpose(1, 2)
        u_Q = u_Q.repeat(1, n_heads, 1, 1)
        u_K = u_K.repeat(1, n_heads, 1, 1)
        u_V = u_V.repeat(1, n_heads, 1, 1)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        scores = torch.matmul(Q, K.transpose(-1, -2))/np.sqrt(d_k) + torch.matmul(u_Q, u_K.transpose(-1, -2))/np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V + u_V)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        output = self.dropout(output)
        enc_outputs = nn.LayerNorm(d_model).cuda()(output + residual)

        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, src_emb_weight, user_emb_weight):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding.from_pretrained(src_emb_weight, freeze=True)
        self.user_emb = nn.Embedding.from_pretrained(user_emb_weight, freeze=True)
        self.dropout = nn.Dropout(p=0.2)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs, u_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.dropout(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        u_outputs = self.user_emb(u_inputs)
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask, u_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs).cuda()  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda()  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).cuda()  # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, src_emb_weight, user_emb_weight):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_emb_weight=src_emb_weight, user_emb_weight=user_emb_weight).cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()

    def forward(self, enc_inputs, dec_inputs, u_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, u_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


model = Transformer(src_emb_weight=src_emb_weight, user_emb_weight=user_emb_weight).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=1e-5)

model.train()

for epoch in range(1):
    for step, (enc_inputs, dec_inputs, dec_outputs, u_inputs) in enumerate(loader):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        '''
        enc_inputs, dec_inputs, dec_outputs, u_inputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda(), u_inputs.cuda()
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs, u_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))
        if step % 20 == (20 - 1):
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def beam_search_decoder(model, num_beams, max_len, enc_input, start_symbol, u_input):
    """
    a beam search implementation about seq2seq with attention
    :param decoder:
    :param num_beams: number of beam, int
    :param max_len: max length of result
    :param input: input of decoder
    :return: list of index
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input, u_input)
    beams = [[start_symbol]]
    b_scores = [1]
    sfm = nn.Softmax(dim=-1)

    for i in range(max_len):
        results = []
        r_scores = []
        for beam, score in zip(beams, b_scores):  # range num is num_beams
            beam = list(beam)
            dec_input = torch.LongTensor(beam).unsqueeze(0).detach().cuda()
            dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
            projected = model.projection(dec_outputs)
            new_gen = projected.squeeze(0)[-1]
            new_gen = sfm(new_gen)[-1]
            M, N = torch.topk(new_gen, k=num_beams)
            for m, n in zip(M, N):
                if n.item() in [E, S, P]:
                    continue
                else:
                    results.append(beam + [n.item()])
                    r_scores.append(score * m.item())

        beams = np.array(results)
        b_scores = np.array(r_scores)
        if len(beams) > num_beams:
            pros_idx = (-1 * b_scores).argsort()[0:num_beams]
            beams = beams[pros_idx]
            b_scores = b_scores[pros_idx]

    return beams, b_scores


model.eval()
tgt_len = 4
num_beams = 10
para = {}
beam_dict = {}
score_dict = {}


with torch.no_grad():
    enc_inputs = torch.LongTensor(src_mtx).cuda()
    u_inputs = torch.LongTensor(np.array(list(range(user_num))).reshape(-1, 1)).cuda()
    for i in range(len(enc_inputs)):
        beams, scores = beam_search_decoder(model, num_beams, tgt_len, enc_inputs[i].view(1, -1), S, u_inputs[i].view(1, -1))
        beam_dict[i] = beams
        score_dict[i] = scores
        if (i + 1) % 1000 == 0:
            print(i + 1)

para['beam_dict'] = beam_dict
para['score_dict'] = score_dict
with open('result.pkl', 'wb') as file:
    pickle.dump(para, file)

print('ending!')
