# coding: utf-8

import json
import nltk
import torch
import functools
import numpy as np
from config import Args
from torch.utils.data import Dataset, DataLoader
from utils import get_wikisql_tables_path, get_preprocess_path, UNK_WORD, get_anonymous_path
from utils import load_data, build_vocab, build_all_vocab, change2idx, pad, max_len_of_m_lists, load_anonymous_data, EOS_WORD


class BindingDataset(Dataset):
    def __init__(self, mode, args, data_from_train=None):
        self.args = args
        device = self.args.device
        # load data
        data_path = get_anonymous_path(mode)
        anony_ques, anony_ques_len, anony_query, anony_query_len = load_anonymous_data(data_path)
        # get len
        self.len = len(anony_ques)
        print('mode {} length: {}'.format(mode, str(self.len)))
        # the data that need use train's data for dev and test
        if data_from_train is None:
            self.anony_ques_max_len, self.anony_query_max_len = max(anony_ques_len), max(anony_query_len)
            anony_ques.extend(anony_query)
            self.anony_ques_vocab, _ = build_vocab(anony_ques, init_vocab={UNK_WORD: 0})
            # init_vocab = {UNK_WORD: 0}
            # for i in range(45 - 1):
            #     key = 'col_' + str(i)
            #     value = len(init_vocab)
            #     init_vocab[key] = value
            # # for i in range(self.args.cells_split_marker_max_len - 1):
            # for i in range(9 - 1):
            #     key = 'val_' + str(i)
            #     value = len(init_vocab)
            #     init_vocab[key] = value
            self.anony_query_vocab = self.anony_ques_vocab
        else:
            self.anony_ques_max_len, self.anony_query_max_len, self.anony_ques_vocab, self.anony_query_vocab = data_from_train
        # change2tensor
        self.anony_ques_tensor = torch.LongTensor(pad(change2idx(anony_ques, vocab=self.anony_ques_vocab, name='anony_ques_'+mode), max_len=self.anony_ques_max_len, pad_token=self.anony_ques_vocab[EOS_WORD])).to(device)
        self.anony_query_tensor = torch.LongTensor(pad(change2idx(anony_query, vocab=self.anony_query_vocab, name='anony_query_'+mode), max_len=self.anony_query_max_len, pad_token=self.anony_query_vocab[EOS_WORD])).to(device)

    def __getitem__(self, index):
        return self.anony_ques_tensor[index], self.anony_query_tensor[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    pass
