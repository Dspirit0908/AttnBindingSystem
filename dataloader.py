# coding: utf-8

import json
import nltk
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import get_wikisql_tables_path, get_preprocess_path
from utils import load_data, load_tables, build_vocab, build_all_vocab, change2idx, pad, max_len_of_m_lists


class BindingDataset(Dataset):
    def __init__(self, mode, only_label, vocab, data_from_train=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        # get path
        data_path, tables_path = get_preprocess_path(mode), get_wikisql_tables_path(mode)
        # load data
        tokenize_list, tokenize_len_list, pos_tag_list, table_id_list, label_list = load_data(data_path, only_label=only_label)
        # get len
        self.len = len(tokenize_list)
        # read tables
        tables_info = load_tables(get_wikisql_tables_path(mode))
        # get columns
        columns_split_list, columns_split_len_list, columns_split_marker_list, columns_split_marker_len_list = [], [], [], []
        for table_id in table_id_list:
            columns_split = tables_info[table_id]['columns_split']
            columns_split_list.append(columns_split)
            columns_split_len_list.append(len(columns_split))

            columns_split_marker = tables_info[table_id]['columns_split_marker']
            columns_split_marker_list.append(columns_split_marker)
            columns_split_marker_len_list.append(len(columns_split_marker))
        # check
        assert len(columns_split_list) == len(columns_split_marker_list) == self.len
        # the data that need use train's data for dev and test
        if mode == 'train':
            self.tokenize_max_len, self.column_token_max_len, self.columns_split_marker_max_len =\
                max(tokenize_len_list), max(columns_split_len_list), max(columns_split_marker_len_list)
            self.pos_tag_vocab, _ = build_vocab(pos_tag_list, init_vocab={'UNK': 0})
        else:
            self.tokenize_max_len, self.column_token_max_len, self.columns_split_marker_max_len, self.pos_tag_vocab = data_from_train
        # get labels
        pointer_label_list = []
        for label in label_list:
            pointer_label = []
            for index, single_label in enumerate(label):
                if single_label == 'UNK':
                    pointer_label.append(self.tokenize_max_len + self.columns_split_marker_max_len)
                    # pointer_label.append(-100)
                else:
                    single_label_split = single_label.split('_')
                    if single_label_split[0] == 'Value':
                        pointer_label.append(int(single_label_split[1]))
                    elif single_label_split[0] == 'Column':
                        pointer_label.append(self.tokenize_max_len + int(single_label_split[1]))
            pointer_label_list.append(pointer_label)
        # change2tensor
        self.tokenize_tensor = torch.LongTensor(pad(change2idx(tokenize_list, vocab=vocab), max_len=self.tokenize_max_len), device=device)
        self.tokenize_len_tensor = torch.LongTensor(list(map(lambda len: min(len, self.tokenize_max_len), tokenize_len_list)), device=device)
        self.pos_tag_tensor = torch.LongTensor(pad(change2idx(pos_tag_list, vocab=self.pos_tag_vocab), max_len=self.tokenize_max_len), device=device)
        self.columns_split_tensor = torch.LongTensor(pad(change2idx(columns_split_list, vocab=vocab), max_len=self.column_token_max_len), device=device)
        self.columns_split_len_tensor = torch.LongTensor(list(map(lambda len: min(len, self.column_token_max_len), columns_split_len_list)), device=device)
        self.columns_split_marker_tensor = torch.LongTensor(pad(columns_split_marker_list, max_len=self.columns_split_marker_max_len, pad_token=self.column_token_max_len), device=device)
        self.columns_split_marker_len_tensor =torch.LongTensor(list(map(lambda len: min(len, self.columns_split_marker_max_len), columns_split_marker_len_list)), device=device)
        self.pointer_label_tensor = torch.LongTensor(pad(pointer_label_list, max_len=self.tokenize_max_len, pad_token=-100), device=device)

    def __getitem__(self, index):
        return (
                    [self.tokenize_tensor[index], self.tokenize_len_tensor[index]],
                    [self.pos_tag_tensor[index], ],
                    [self.columns_split_tensor[index], self.columns_split_len_tensor[index]],
                    [self.columns_split_marker_tensor[index], self.columns_split_marker_len_tensor[index]],
                ), self.pointer_label_tensor[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    word2index, index2word = build_all_vocab()
    data_from_train = None
    mode_list = ['train', 'dev', 'test']
    for mode in mode_list:
        if mode == 'train':
            dataset = BindingDataset(mode, only_label=True, vocab=word2index)
            train_dataloader = DataLoader(dataset=dataset, batch_size=32)
            data_from_train = (dataset.tokenize_max_len, dataset.column_token_max_len, dataset.columns_split_marker_max_len, dataset.pos_tag_vocab)
        else:
            dataset = BindingDataset(mode, only_label=True, vocab=word2index, data_from_train=data_from_train)
            dataloader = DataLoader(dataset=dataset, batch_size=32)
