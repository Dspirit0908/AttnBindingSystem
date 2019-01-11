# coding: utf-8

import json
import nltk
import torch
import functools
import numpy as np
from config import Args
from torch.utils.data import Dataset, DataLoader
from utils import get_wikisql_tables_path, get_preprocess_path, UNK_WORD
from utils import load_data, build_vocab, build_all_vocab, change2idx, pad, max_len_of_m_lists


class BindingDataset(Dataset):
    def __init__(self, mode, args, data_from_train=None):
        self.args = args
        device = self.args.device
        # get path
        data_path, tables_path = get_preprocess_path(mode), get_wikisql_tables_path(mode)
        # load data
        tokenize_list, tokenize_len_list, pos_tag_list, table_id_list,\
        (columns_split_list, columns_split_len_list, columns_split_marker_list, columns_split_marker_len_list),\
        (cells_split_list, cells_split_len_list, cells_split_marker_list, cells_split_marker_len_list),\
        label_list, sql_sel_col_list, sql_conds_cols_list, sql_conds_values_list = load_data(data_path, only_label=self.args.only_label)
        # get len
        self.len = len(tokenize_list)
        # the data that need use train's data for dev and test
        if data_from_train is None:
            self.tokenize_max_len, self.columns_token_max_len, self.columns_split_marker_max_len, self.cells_token_max_len, self.cells_split_marker_max_len =\
                max(tokenize_len_list), max(columns_split_len_list), max(columns_split_marker_len_list), max(cells_split_len_list), max(cells_split_marker_len_list)
            self.pos_tag_vocab, _ = build_vocab(pos_tag_list, init_vocab={UNK_WORD: 0})
        else:
            self.tokenize_max_len, self.columns_token_max_len, self.columns_split_marker_max_len, self.cells_token_max_len, self.cells_split_marker_max_len, self.pos_tag_vocab = data_from_train

        # get labels
        def _get_label(model, pos, index, suffix=None):
            if model == 'gate':
                if pos == 0: return 0
                if pos == 1: return 1 + suffix
                if pos == 2: return 1 + self.columns_split_marker_max_len - 1 + suffix if self.args.cell_info else 1 + self.columns_split_marker_max_len - 1
            elif model == 'baseline':
                if pos == 0: return index
                if pos == 1: return self.tokenize_max_len + suffix
                if pos == 2: return self.tokenize_max_len + self.columns_split_marker_max_len - 1 + suffix
        _get_label = functools.partial(_get_label, model=self.args.model)
        pointer_label_list, gate_label_list = [], []
        for label in label_list:
            pointer_label, gate_label = [], []
            for index, single_label in enumerate(label):
                if single_label == UNK_WORD:
                    pointer_label.append(_get_label(pos=0, index=index))
                    gate_label.append(0)
                else:
                    single_label_split = single_label.split('_')
                    if single_label_split[0] == 'Column':
                        pointer_label.append(_get_label(pos=1, index=index, suffix=int(single_label_split[1])))
                        gate_label.append(1)
                    elif single_label_split[0] == 'Value':
                        pointer_label.append(_get_label(pos=2, index=index, suffix=int(single_label_split[1])))
                        gate_label.append(2)
            pointer_label_list.append(pointer_label), gate_label_list.append(gate_label)
        # change2tensor
        self.tokenize_tensor = torch.LongTensor(pad(change2idx(tokenize_list, vocab=self.args.vocab, name='tokenize_'+mode), max_len=self.tokenize_max_len)).to(device)
        self.tokenize_len_tensor = torch.LongTensor(list(map(lambda len: min(len, self.tokenize_max_len), tokenize_len_list))).to(device)
        self.pos_tag_tensor = torch.LongTensor(pad(change2idx(pos_tag_list, vocab=self.pos_tag_vocab, name='pos_tag_'+mode), max_len=self.tokenize_max_len)).to(device)
        self.columns_split_tensor = torch.LongTensor(pad(change2idx(columns_split_list, vocab=self.args.vocab, name='columns_split_'+mode), max_len=self.columns_token_max_len)).to(device)
        self.columns_split_len_tensor = torch.LongTensor(list(map(lambda len: min(len, self.columns_token_max_len), columns_split_len_list))).to(device)
        self.columns_split_marker_tensor = torch.LongTensor(pad(columns_split_marker_list, max_len=self.columns_split_marker_max_len, pad_token=self.columns_token_max_len - 1)).to(device)
        self.columns_split_marker_len_tensor = torch.LongTensor(list(map(lambda len: min(len, self.columns_split_marker_max_len), columns_split_marker_len_list))).to(device)
        self.cells_split_tensor = torch.LongTensor(pad(change2idx(cells_split_list, vocab=self.args.vocab, name='cells_split_' + mode), max_len=self.cells_token_max_len)).to(device)
        self.cells_split_len_tensor = torch.LongTensor(list(map(lambda len: min(len, self.cells_token_max_len), cells_split_len_list))).to(device)
        self.cells_split_marker_tensor = torch.LongTensor(pad(cells_split_marker_list, max_len=self.cells_split_marker_max_len, pad_token=self.cells_token_max_len - 1)).to(device)
        self.cells_split_marker_len_tensor = torch.LongTensor(list(map(lambda len: min(len, self.cells_split_marker_max_len), cells_split_marker_len_list))).to(device)
        # can not pad -100 for crf
        if args.crf:
            pad_token = 0
        else:
            pad_token = -100
        self.pointer_label_tensor = torch.LongTensor(pad(pointer_label_list, max_len=self.tokenize_max_len, pad_token=pad_token)).to(device)
        self.gate_label_tensor = torch.LongTensor(pad(gate_label_list, max_len=self.tokenize_max_len, pad_token=-100)).to(device)
        # handle sql_sel_col_list, sql_conds_cols_list, sql_conds_values_list
        self.sql_sel_col_list = torch.LongTensor(sql_sel_col_list)
        assert max_len_of_m_lists(sql_conds_cols_list) == max_len_of_m_lists(sql_conds_values_list)
        self.sql_conds_cols_list = torch.LongTensor(pad(sql_conds_cols_list, max_len=max_len_of_m_lists(sql_conds_cols_list), pad_token=-100))
        self.sql_conds_values_list = torch.LongTensor(pad(sql_conds_values_list, max_len=max_len_of_m_lists(sql_conds_values_list), pad_token=[-100, -100]))

    def __getitem__(self, index):
        return (
                    [self.tokenize_tensor[index], self.tokenize_len_tensor[index]],
                    [self.pos_tag_tensor[index], ],
                    [self.columns_split_tensor[index], self.columns_split_len_tensor[index]],
                    [self.columns_split_marker_tensor[index], self.columns_split_marker_len_tensor[index]],
                    [self.cells_split_tensor[index], self.cells_split_len_tensor[index]],
                    [self.cells_split_marker_tensor[index], self.cells_split_marker_len_tensor[index]],
                ),\
                    (self.pointer_label_tensor[index], self.gate_label_tensor[index]),\
                    (self.sql_sel_col_list[index], self.sql_conds_cols_list[index], self.sql_conds_values_list[index])

    def __len__(self):
        return self.len


if __name__ == '__main__':
    args = Args()
    word2index, index2word = build_all_vocab(init_vocab={UNK_WORD: 0})
    args.vocab, args.vocab_size = word2index, len(word2index)
    print(args.vocab_size)
    args.model = 'baseline'
    train_dataset = BindingDataset('train', args=args)
