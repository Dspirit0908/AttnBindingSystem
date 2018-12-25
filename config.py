# coding: utf-8

import torch

data_path = './data/'
wikisql_path = data_path + 'wikisql/'
preprocess_path = data_path + 'preprocess/'
word_embedding_path = data_path + 'glove.6B.300d.txt'


# hyperparameters
class Args(object):
    def __init__(self):
        self.seed = 72
        self.lr = 0.001
        self.weight_decay = 0.001
        self.epochs = 100
        self.batch_size = 256
        self.log_trian_interval = 10
        self.log_test_interval = 1
        self.dropout_p = 0.1
        self.load_cell = False
        self.load_w2v = False
        self.only_label = False
        self.shuffle = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda = True if str(self.device) == 'cuda' else False
        self.word_dim = 100
        self.hidden_size = 100
        self.num_layers = 1
        self.teacher_forcing_ratio = 1.0
        self.embed_matrix = None


if __name__ == '__main__':
    args = Args()
    print(args.cuda)
