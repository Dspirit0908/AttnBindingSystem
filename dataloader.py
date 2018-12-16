# coding: utf-8

import json
import nltk
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import get_wikisql_tables_path, get_preprocess_path
from utils import load_data, load_tables, build_vocab


class BindingDataset(Dataset):
    def __init__(self, mode, only_label):
        # get path
        data_path, tables_path = get_preprocess_path(mode), get_wikisql_tables_path(mode)
        # load data
        tokenize_list, tokenize_len_list, pos_tag_list, table_id_list, label_list = load_data(data_path, only_label=only_label)
        # get len
        self.len = len(tokenize_list)
        # read tables
        tables_info = load_tables(get_wikisql_tables_path(mode))
        # get columns
        columns_split_list, columns_split_marker_list = [], []
        for table_id in table_id_list:
            columns_split_list.append(tables_info[table_id]['columns_split'])
            columns_split_marker_list.append(tables_info[table_id]['columns_split_marker'])
        # check
        assert len(columns_split_list) == len(columns_split_marker_list) == self.len
        # get labels

        # change to index

        # pad (truncate?)

        

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.len


if __name__ == '__main__':
    mode_list = ['train', 'dev', 'test']
    for mode in mode_list:
        dataset = BindingDataset(mode, only_label=True)
