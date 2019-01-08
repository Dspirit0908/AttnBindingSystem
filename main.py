# coding: utf-8

import os
import torch
import logging
import functools
from config import Args
from train import train, eval, train_rl, eval_rl, test
from models.gate import Gate
from models.baseline import Baseline
from dataloader import BindingDataset
from torch.utils.data import Dataset, DataLoader
from utils import UNK_WORD, BOS_WORD, build_all_vocab, set_seed, load_word_embedding, add_abstraction


def main(mode, model='baseline'):
    # set environ, args, seed, loggging
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    args = Args()
    args.model = model
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('binding')
    # build vocab
    word2index, index2word = build_all_vocab(init_vocab={UNK_WORD: 0, BOS_WORD: 1})
    args.vocab, args.vocab_size, args.index2word = word2index, len(word2index), index2word
    # get data_from_train from only_label = True, for same as train baseline
    args.only_label = True
    train_dataset = BindingDataset('train', args=args)
    data_from_train = (train_dataset.tokenize_max_len, train_dataset.columns_token_max_len,
                       train_dataset.columns_split_marker_max_len, train_dataset.cells_token_max_len,
                       train_dataset.cells_split_marker_max_len, train_dataset.pos_tag_vocab)
    args.tokenize_max_len, args.columns_token_max_len, args.columns_split_marker_max_len, \
    args.cells_token_max_len, args.cells_split_marker_max_len, args.pos_tag_vocab = data_from_train
    logger.info('data_from_train'), logger.info(data_from_train)
    # set only_label
    if mode == 'train baseline':
        args.only_label = True
    elif mode == 'policy gradient':
        args.only_label = False
    elif mode == 'test model':
        args.only_label = True
    elif mode == 'add feature':
        args.only_label = False
    # build train_dataloader
    train_dataset = BindingDataset('train', args=args, data_from_train=data_from_train)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    # build dev_dataloader
    args.shuffle = False
    dev_dataset = BindingDataset('dev', args=args, data_from_train=data_from_train)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    # build test_dataloader
    test_dataset = BindingDataset('test', args=args, data_from_train=data_from_train)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    # load word embedding
    if args.load_w2v:
        args.embed_matrix = load_word_embedding(args.word_dim, word2index)
    # train
    if mode == 'train baseline':
        if args.model == 'baseline':
            model = Baseline(args)
        elif args.model == 'gate':
            model = Gate(args)
        train(train_dataloader, dev_dataloader, args=args, model=model)
    elif mode == 'policy gradient':
        model = torch.load('./res/' + args.model + '/2705', map_location=lambda storage, loc: storage.cuda(0))
        train_rl(train_dataloader, dev_dataloader, args=args, model=model)
    elif mode == 'test model':
        # also need the correct 'model' for dataloader
        model = torch.load('./res/gate/2705', map_location=lambda storage, loc: storage.cuda(0))
        eval(dev_dataloader, args, model, epoch=0)
        eval_rl(dev_dataloader, args, model, epoch=0)
    elif mode == 'add feature':
        model = torch.load('./res/policy_gradient/0.804922_22-16-28', map_location=lambda storage, loc: storage.cuda(0))
        res = test(test_dataloader, args, model)
        add_abstraction('test', res=res, args=args)


if __name__ == '__main__':
    # main('train baseline', 'gate')
    main('test model', 'gate')
    # main('policy gradient', 'gate')
    # main('add feature', 'gate')
