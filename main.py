# coding: utf-8

import torch
import functools
from config import Args
from dataloader import BindingDataset
from torch.utils.data import Dataset, DataLoader
from utils import get_preprocess_path, get_wikisql_tables_path, load_data, load_tables, build_vocab, build_all_vocab


def main():
    args = Args()
    # build vocab
    word2index, index2word = build_all_vocab()
    # build dataloader
    train_dataset = BindingDataset('train', only_label=True, vocab=word2index)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    data_from_train = (train_dataset.tokenize_max_len, train_dataset.column_token_max_len,
                       train_dataset.columns_split_marker_max_len, train_dataset.pos_tag_vocab)
    dev_dataset = BindingDataset('dev', only_label=True, vocab=word2index, data_from_train=data_from_train)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    # load word embedding
    args.embed_matrix = None



if __name__ == '__main__':
    main()
