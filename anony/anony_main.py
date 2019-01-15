# coding: utf-8

import torch
import logging
from config import Args
from anony.anony_dataloader import BindingDataset
from torch.utils.data import DataLoader
from models.seq2seq import Encoder, Decoder, Seq2Seq
from anony.anony_train import train
from utils import set_seed


def main(args):
    train_dataset = BindingDataset('train', args=args)
    data_from_train = train_dataset.anony_ques_max_len, train_dataset.anony_query_max_len, train_dataset.anony_ques_vocab, train_dataset.anony_query_vocab
    args.anony_ques_max_len, args.anony_query_max_len, args.anony_ques_vocab, args.anony_query_vocab = data_from_train
    print(args.anony_ques_max_len, args.anony_query_max_len, len(args.anony_ques_vocab), args.anony_query_vocab)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    # build dev_dataloader
    args.shuffle = False
    dev_dataset = BindingDataset('dev', args=args, data_from_train=data_from_train)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    # build test_dataloader
    # test_dataset = BindingDataset('test', args=args, data_from_train=data_from_train)
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    # load word embedding
    # train
    encoder = Encoder(len(args.anony_ques_vocab), args.word_dim, args.hidden_size,
                      n_layers=2 * args.num_layers, dropout=args.dropout_p)
    decoder = Decoder(args.word_dim, args.hidden_size, len(args.anony_query_vocab),
                      n_layers=args.num_layers, dropout=args.dropout_p)
    model = Seq2Seq(encoder, decoder)
    train(train_dataloader, dev_dataloader, args, model)


if __name__ == '__main__':
    # set environ, loggging
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    torch.cuda.set_device(3)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('binding')
    # set args
    args = Args()
    set_seed(args.seed)
    logger.info(args.device)
    main(args)
