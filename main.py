# coding: utf-8

import os
import torch
import logging
import functools
from config import Args
from train import train, train_rl
from models.baseline import Baseline
from dataloader import BindingDataset
from torch.utils.data import Dataset, DataLoader
from utils import build_all_vocab, set_seed, load_word_embedding


def main(mode):
    # set environ, args, seed, loggging
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    # torch.cuda.set_device(4)
    args = Args()
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('binding')
    # build vocab
    word2index, index2word = build_all_vocab(init_vocab={'UNK': 0})
    args.vocab, args.vocab_size = word2index, len(word2index)
    # get data_from_train from only_label = False, for same as train baseline
    args.only_label = True
    train_dataset = BindingDataset('train', args=args)
    data_from_train = (train_dataset.tokenize_max_len, train_dataset.columns_token_max_len,
                       train_dataset.columns_split_marker_max_len, train_dataset.cells_token_max_len,
                       train_dataset.cells_split_marker_max_len, train_dataset.pos_tag_vocab)
    args.tokenize_max_len, args.columns_token_max_len, args.columns_split_marker_max_len, \
    args.cells_token_max_len, args.cells_split_marker_max_len, args.pos_tag_vocab = data_from_train
    logger.info(data_from_train)
    args.only_label = True if mode == 'train baseline' else False
    # build train_dataloader
    train_dataset = BindingDataset('train', args=args, data_from_train=data_from_train)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    # build dev_dataloader
    dev_dataset = BindingDataset('dev', args=args, data_from_train=data_from_train)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    # load word embedding
    if args.load_w2v:
        args.embed_matrix = load_word_embedding(args.word_dim, word2index)
    # train
    if mode == 'train baseline':
        model = Baseline(args)
        train(train_dataloader, dev_dataloader, args=args, model=model)
    elif mode == 'policy gradient':
        model = torch.load('./res/1516')
        train_rl(train_dataloader, dev_dataloader, args=args, model=model)


if __name__ == '__main__':
    main('policy gradient')
