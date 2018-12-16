# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
from models.modules.utils import encode_unsorted_batch
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class TableRNNEncoder(nn.Module):
    def __init__(self, encoder, split_type='incell', merge_type='cat'):
        super(TableRNNEncoder, self).__init__()
        self.split_type = split_type
        self.merge_type = merge_type
        self.hidden_size = encoder.hidden_size
        self.encoder = encoder
        if self.merge_type == 'mlp':
            self.merge = nn.Sequential(
                nn.Linear(2 * self.hidden_size, self.hidden_size),
                nn.Tanh())

    def forward(self, tbl, tbl_len, tbl_split):
        """
        Encode table headers.
            :param tbl: header token list
            :param tbl_len: length of token list (num_table_header, batch)
            :param tbl_split: table header boundary list
        """
        tbl_context = encode_unsorted_batch(self.encoder, tbl, tbl_len)
        # --> (num_table_header, batch, hidden_size * num_directions)
        if self.split_type == 'outcell':
            batch_index = torch.LongTensor(range(tbl_split.data.size(1))).unsqueeze_(
                0).cuda().expand_as(tbl_split.data)
            enc_split = tbl_context[tbl_split.data, batch_index, :]
            enc_left, enc_right = enc_split[:-1], enc_split[1:]
        elif self.split_type == 'incell':
            batch_index = torch.LongTensor(range(tbl_split.data.size(1))).unsqueeze_(
                0).cuda().expand(tbl_split.data.size(0) - 1, tbl_split.data.size(1))
            split_left = (tbl_split.data[:-1] +
                          1).clamp(0, tbl_context.size(0) - 1)
            enc_left = tbl_context[split_left, batch_index, :]
            split_right = (tbl_split.data[1:] -
                           1).clamp(0, tbl_context.size(0) - 1)
            enc_right = tbl_context[split_right, batch_index, :]

        if self.merge_type == 'sub':
            return (enc_right - enc_left)
        elif self.merge_type == 'cat':
            # take half vector for each direction
            half_hidden_size = self.hidden_size // 2
            return torch.cat([enc_right[:, :, :half_hidden_size], enc_left[:, :, half_hidden_size:]], 2)
        elif self.merge_type == 'mlp':
            return self.merge(torch.cat([enc_right, enc_left], 2))
