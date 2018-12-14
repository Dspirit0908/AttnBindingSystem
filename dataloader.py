# coding: utf-8

import json
import nltk
import torch
import numpy as np
from torch.utils.data import Dataset
from config import wikisql_path, preprocess_path
from utils import load_data, build_vocab


class BindingDataset(Dataset):
    def __init__(self, mode, vocab=None):
        data_path, tables_path = preprocess_path + mode + '.jsonl', wikisql_path + mode + '.tables.jsonl'

        tokenize_list, tokenize_len_list, pos_tag_list = load_data(data_path)
                
        # check
        assert len(tokenize_list) == len(tokenize_len_list) == len(pos_tag_list)
        self.len = len(tokenize_list)
        
        # get vocab
        
        # pad (truncate?)

        

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.len


if __name__ == '__main__':
    mode_list = ['train', 'dev', 'test']
    for mode in mode_list:
        dataset = BindingDataset(mode)
