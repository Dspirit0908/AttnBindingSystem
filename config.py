# coding: utf-8

import os
import torch

data_path = './data/'
wikisql_path = data_path + 'wikisql/'
preprocess_path = data_path + 'preprocess/'
anonymous_path = data_path + 'anonymous/'
bert_path = data_path + 'bert/'
word_embedding_path = data_path + 'glove.6B.300d.txt'


# hyperparameters
class Args(object):
    def __init__(self):
        self.seed = 72
        self.lr = 1e-3
        self.small_lr = 1e-5
        self.weight_decay = 0.001
        self.epochs = 200
        self.batch_size = 32
        self.log_trian_interval = 10
        self.log_test_interval = 1
        self.num_layers = 1
        self.dropout_p = 0.1
        self.teacher_forcing_ratio = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda = True if str(self.device) == 'cuda' else False
        self.save_bar_pretrained = 0.45
        self.save_bar_rl = 0.7
        self.gate_class = 3
        self.shuffle = True
        self.only_label = False
        self.load_w2v = False
        self.word_dim = 100
        self.embed_matrix = None
        self.hidden_size = 100
        self.m = 10
        self.model = 'baseline'
        self.bert_model = 'bert-base-uncased'
        self.cell_info = False
        self.attn_concat = False
        self.crf = False


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.backends.cudnn.version())
    args = Args()
    print(args.device, args.cuda)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_emb', action='store_true',
                        help='Use trained word embedding for SQLNet.')
    args = parser.parse_args()
    print(args.train_emb)
