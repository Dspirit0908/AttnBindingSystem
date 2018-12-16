# coding: utf-8

import torch
from torch import nn
from torchcrf import CRF
from binding.utils import runBiRNN, sequence_mask
from binding.config import more_feature_num
from sklearn.metrics import confusion_matrix


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.tokenFeatDim = len(args.tokenFeatDict)
        self.posTagsDim = len(args.posTagsDict)

        self.embedding = nn.Embedding(args.vocab_size, args.word_dim)
        if args.embed_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(args.embed_matrix))
        self.bi_lstm = nn.LSTM(args.word_dim + self.tokenFeatDim + more_feature_num + self.posTagsDim, args.hidden_size, bidirectional=True, batch_first=True,
                               num_layers=args.num_layers, dropout=args.dropout_p)
        self.crf = CRF(args.class_num)
        self.dropout = nn.Dropout(args.dropout_p)
        self.fc = nn.Linear(2 * args.hidden_size, args.class_num)

        self.fc_col = nn.Linear(args.max_len * args.class_num, args.col_num)

        # self.bi_lstm = nn.LSTM(args.hidden_size, args.hidden_size,
        #                        bidirectional=True, batch_first=True,
        #                        num_layers=args.num_layers, dropout=args.dropout_p)
        # self.w_fc = nn.Linear(args.word_dim, args.hidden_size, bias=False)
        # self.u_fc = nn.Linear(self.tokenFeatDim + more_feature_num + self.posTagsDim, args.hidden_size, bias=False)
        # self.v_fc = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

    def forward(self, inputs, labels):
        questions, q_lens = inputs[0]
        q_embed = self.embedding(questions)  # (B, L_Q, D)
        batch_size = q_embed.size(0)
        type = inputs[1][0]
        pos_tags = inputs[2][0]
        col_num_labels = inputs[3][0]
        sql_labels = inputs[4][0]
        # (B, L_Q)
        flags = inputs[4][1]
        # (B)

        q_embed = torch.cat([q_embed, type, pos_tags], 2)
        q_out, hidden = runBiRNN(self.bi_lstm, q_embed, q_lens, total_length=self.args.max_len)
        # (B, L_Q, 2H) Bi-LSTM
        q_out = q_out.transpose(0, 1)
        # (L_Q, B, 2H)
        q_out = self.fc(q_out)
        # (L_Q, B, C)

        # q_col = self.fc_col(q_out.transpose(0, 1).contiguous().view(batch_size, -1))
        # col_loss = nn.functional.cross_entropy(q_col, col_num_labels)
        sql_loss = nn.functional.cross_entropy(q_out.transpose(0, 1).transpose(1, 2), sql_labels, ignore_index=-100)
        l_loss = nn.functional.cross_entropy(q_out.transpose(0, 1).transpose(1, 2), labels, ignore_index=-100)
        return sql_loss