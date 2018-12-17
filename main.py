# coding: utf-8

import os
import torch
import functools
from config import Args
from train import train
from models.model import Model
from dataloader import BindingDataset
from torch.utils.data import Dataset, DataLoader
from utils import build_all_vocab, set_seed, max_len_of_m_lists


def main():
    # set environ, args, seed
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    torch.cuda.set_device(4)
    args = Args()
    set_seed(args.seed)
    # build vocab
    word2index, index2word = build_all_vocab()
    args.vocab_size = len(word2index)
    # build train_dataloader
    train_dataset = BindingDataset('train', only_label=True, vocab=word2index)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    data_from_train = (train_dataset.tokenize_max_len, train_dataset.column_token_max_len,
                       train_dataset.columns_split_marker_max_len, train_dataset.pos_tag_vocab)
    args.tokenize_max_len, args.column_token_max_len, args.columns_split_marker_max_len, args.pos_tag_vocab = \
        train_dataset.tokenize_max_len, train_dataset.column_token_max_len, train_dataset.columns_split_marker_max_len, train_dataset.pos_tag_vocab
    # build dev_dataloader
    train_dataset = BindingDataset('dev', only_label=True, vocab=word2index, data_from_train=data_from_train)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    dev_dataset = BindingDataset('test', only_label=True, vocab=word2index, data_from_train=data_from_train)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    # load word embedding
    args.embed_matrix = None
    # train
    model = Model(args)
    train(train_dataloader, dev_dataloader, args=args, model=model)


if __name__ == '__main__':
    main()
