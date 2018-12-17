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
        self.lstm = nn.LSTM(2*args.hidden_size, args.hidden_size, bidirectional=True, batch_first=False,
                                  num_layers=args.num_layers, dropout=args.dropout_p)
        self.attention = Attention('general', dim=2 * self.args.hidden_size, args=self.args)
        
    def forward(self, memory_bank, tgt, hidden, src_lengths=None, src_max_len=None):
        def _fix_enc_hidden(h):
            """
            The encoder hidden is  (layers*directions) x batch x dim.
            We need to convert it to layers x batch x (directions*dim).
            """
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h
        
        # RNN
        rnn_output, hidden = self.lstm(tgt, hidden)
        # Attention
        rnn_output = rnn_output.transpose(0, 1).contiguous()
        attn_h, align_score = self.attention(memory_bank, rnn_output, src_lengths=src_lengths, src_max_len=src_max_len)
        return align_score, hidden
