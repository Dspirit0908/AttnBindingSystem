# coding: utf-8

import torch
from torch import nn
from torchcrf import CRF
from utils import runBiRNN, sequence_mask
from models.modules.TableRNNEncoder import TableRNNEncoder
from models.modules.Attention import Attention
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
        self.table_encoder = TableRNNEncoder(self.table_bilstm, self.args)
        # question-table attention
        self.ques_table_attn = Attention('general', dim=2*self.hidden_size, args=self.args)
        # top_encoder_lstm
        self.top_encoder_lstm = nn.LSTM(2*args.hidden_size, args.hidden_size, bidirectional=True, batch_first=False,
                                  num_layers=args.num_layers, dropout=args.dropout_p)

    def forward(self, inputs):
        # unpack inputs to data
        tokenize, tokenize_len = inputs[0]  # _, (batch_size)
        pos_tag = inputs[1][0]
        columns_split, columns_split_len = inputs[2]
        columns_split_marker, columns_split_marker_len = inputs[3]
        # encode token
        token_embed = self.token_embedding(tokenize).transpose(0, 1)  # (tokenize_max_len, batch_size, word_dim)
        token_out, token_hidden = runBiRNN(self.token_lstm, token_embed, tokenize_len, total_length=self.args.tokenize_max_len)  # (tokenize_max_len, batch_size, 2*hidden_size), _
        # encode table
        table_embed = self.token_embedding(columns_split).transpose(0, 1)  # (column_token_max_len, batch_size, word_dim)
        table_out = self.table_encoder(table_embed, columns_split_len, columns_split_marker)  # (columns_split_marker_max_len-1, batch_size, 2*hidden_size)
        # question-table attention
        token_out, table_out = token_out.transpose(0, 1), table_out.transpose(0, 1)
        attn_h, align_score = self.ques_table_attn(table_out, token_out, tokenize_len, max_len=self.args.tokenize_max_len)  # (batch_size, tokenize_max_len, 2*hidden_size), (batch_size, tokenize_max_len, columns_split_marker_max_len-1)
        # top_encoder_lstm
        attn_h = attn_h.transpose(0, 1)  # (tokenize_max_len, batch_size, 2*hidden_size)
        top_out, top_hidden = runBiRNN(self.top_encoder_lstm, attn_h, tokenize_len, total_length=self.args.tokenize_max_len)  # (tokenize_max_len, batch_size, 2*hidden_size), _
        
        return
