# coding=utf-8

import torch
from torch import nn
from torch.autograd import Variable
from models.modules.Attention import Attention


class PointerNetRNNDecoder(nn.Module):
    """
    Pointer network RNN Decoder, process all the output together
    """
    def __init__(self, args):
        super(PointerNetRNNDecoder, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(args.word_dim, args.hidden_size, bidirectional=True, batch_first=False,
                                  num_layers=args.num_layers, dropout=args.dropout_p)
        self.attention = Attention('general', dim=2 * self.args.hidden_size, args=self.args)
        
    def forward(self, tgt, memory_bank, hidden, memory_lengths=None):
        # RNN
        rnn_output, _ = self.lstm(tgt, hidden)
        # Attention
        rnn_output = rnn_output.transpose(0, 1)
        attn_h, align_score = self.attention(memory_bank, rnn_output, src_lengths=memory_lengths, src_max_len=self.args.tokenize_max_len + self.args.columns_split_marker_max_len - 1)
        return align_score
