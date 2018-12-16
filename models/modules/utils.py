# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def sort_for_pack(input_len):
    idx_sorted, input_len_sorted = zip(
        *sorted(list(enumerate(input_len)), key=lambda x: x[1], reverse=True))
    idx_sorted, input_len_sorted = list(idx_sorted), list(input_len_sorted)
    idx_map_back = list(map(lambda x: x[0], sorted(
        list(enumerate(idx_sorted)), key=lambda x: x[1])))
    return idx_sorted, input_len_sorted, idx_map_back


def encode_unsorted_batch(encoder, tbl, tbl_len):
    # sort for pack()
    idx_sorted, tbl_len_sorted, idx_map_back = sort_for_pack(tbl_len)
    tbl_sorted = tbl.index_select(1, Variable(
        torch.LongTensor(idx_sorted).cuda(), requires_grad=False))
    # tbl_context: (seq_len, batch, hidden_size * num_directions)
    __, tbl_context = encoder(tbl_sorted, tbl_len_sorted)
    # recover the sort for pack()
    v_idx_map_back = Variable(torch.LongTensor(
        idx_map_back).cuda(), requires_grad=False)
    tbl_context = tbl_context.index_select(1, v_idx_map_back)
    return tbl_context
