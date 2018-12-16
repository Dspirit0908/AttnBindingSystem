# coding: utf-8

import torch
from torch import nn
from torchcrf import CRF
from utils import runBiRNN, sequence_mask
from models.modules.TableRNNEncoder import TableRNNEncoder
from sklearn.metrics import confusion_matrix


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # args
        self.args = args
        self.hidden_size = args.hidden_size
        # embedding
        self.token_embedding = nn.Embedding(args.vocab_size, args.word_dim)
        if args.embed_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(args.embed_matrix))
        # token_lstm
        self.token_lstm = nn.LSTM(args.word_dim, args.hidden_size, bidirectional=True, batch_first=False,
                               num_layers=args.num_layers, dropout=args.dropout_p)
        # table_encoder
        self.table_bilstm = nn.LSTM(args.word_dim, args.hidden_size, bidirectional=True, batch_first=False,
                               num_layers=args.num_layers, dropout=args.dropout_p)
        self.table_encoder = TableRNNEncoder(self.table_bilstm)
        # fc
        # self.fc = nn.Linear(2 * args.hidden_size, args.class_num)

    def forward(self, inputs):
        # unpack inputs to data
        tokenize, tokenize_len = inputs[0]
        pos_tag = inputs[1][0]
        columns_split, columns_split_len = inputs[2]
        columns_split_marker, columns_split_marker_len = inputs[3]
        # encode token
        token_embed = self.token_embedding(tokenize).transpose(0, 1)  # (tokenize_max_len, batch_size, word_dim)
        token_out, hidden = runBiRNN(self.token_lstm, token_embed, tokenize_len, total_length=self.args.tokenize_max_len)  # (tokenize_max_len, batch_size, 2*hidden_size), _
        print(token_out.size())
        # encode table
        table_embed = self.table_encoder(columns_split, columns_split_len, columns_split_marker)
        print(table_embed.size())
        return
