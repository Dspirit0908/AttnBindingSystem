# coding: utf-8

import sys
import torch
import random
import logging
import numpy as np
from torch import nn
from torchcrf import CRF
from torch.autograd import Variable
from utils import BOS_WORD, runBiRNN, sequence_mask, fix_hidden
from models.modules.TableRNNEncoder import TableRNNEncoder
from models.modules.GlobalAttention import GlobalAttention
from models.modules.PointerNet import PointerNetRNNDecoder
from models.modules.PointerNetDecoderStep import Decoder
from models.modules.Decoder import AttnDecoderRNN
from allennlp.modules import ConditionalRandomField

logger = logging.getLogger('binding')
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile='full')


class Gate(nn.Module):
    def __init__(self, args):
        super(Gate, self).__init__()
        # args
        self.args = args
        self.hidden_size = args.hidden_size
        # embedding
        self.token_embedding = nn.Embedding(args.vocab_size, args.word_dim)
        if args.embed_matrix is not None:
            self.token_embedding.weight = nn.Parameter(torch.FloatTensor(args.embed_matrix))
        # pos_tag embedding
        self.pos_tag_embedding = nn.Embedding(len(args.pos_tag_vocab), args.word_dim)
        # token_lstm
        self.token_lstm = nn.LSTM(args.word_dim, args.hidden_size, bidirectional=True, batch_first=False,
                               num_layers=args.num_layers, dropout=args.dropout_p)
        self.col_lstm = nn.LSTM(args.word_dim, args.hidden_size, bidirectional=True, batch_first=False,
                                  num_layers=args.num_layers, dropout=args.dropout_p)
        self.cell_lstm = nn.LSTM(args.word_dim, args.hidden_size, bidirectional=True, batch_first=False,
                                  num_layers=args.num_layers, dropout=args.dropout_p)
        # table_encoder
        self.table_encoder = TableRNNEncoder(self.args)
        # gate
        if self.args.cell_info:
            self.gate = nn.Linear(6 * self.args.hidden_size, self.args.gate_class)
        else:
            self.gate = nn.Linear(4 * self.args.hidden_size, self.args.gate_class)
        # col pointer network
        self.col_pointer_network = GlobalAttention(args=self.args, dim=2 * self.args.hidden_size, attn_type="mlp")
        # cell pointer network
        self.cell_pointer_network = GlobalAttention(args=self.args, dim=2 * self.args.hidden_size, attn_type="mlp")
        if self.args.crf:
            # todo: set num for baseline
            self.crf = ConditionalRandomField(46)

    def forward(self, inputs):
        # unpack inputs to data
        tokenize, tokenize_len = inputs[0]  # _, (batch_size)
        pos_tag = inputs[1][0]
        columns_split, columns_split_len = inputs[2]
        columns_split_marker, columns_split_marker_len = inputs[3]  # _, (batch_size)
        cells_split, cells_split_len = inputs[4]
        cells_split_marker, cells_split_marker_len = inputs[5]  # _, (batch_size)
        batch_size = tokenize.size(0)
        # encode token
        token_embed = self.token_embedding(tokenize)
        token_embed = token_embed.transpose(0, 1).contiguous()  # (tokenize_max_len, batch_size, word_dim)
        # add pos_tag on token_embed
        pos_tag_embed = self.pos_tag_embedding(pos_tag).transpose(0, 1).contiguous()  # (tokenize_max_len, batch_size, word_dim)
        token_embed += pos_tag_embed
        # run token lstm; (tokenize_max_len, batch_size, 2 * hidden_size), _
        token_out, token_hidden = runBiRNN(self.token_lstm, token_embed, tokenize_len, total_length=self.args.tokenize_max_len)
        # encode columns
        col_embed = self.token_embedding(columns_split).transpose(0, 1).contiguous()  # (columns_token_max_len, batch_size, word_dim)
        # (columns_split_marker_max_len - 1, batch_size, 2 * hidden_size), ((layer * bidirectional, batch, hidden_size), _)
        col_out, col_hidden = self.table_encoder(self.col_lstm, col_embed, columns_split_len, columns_split_marker, hidden=None, total_length=self.args.columns_token_max_len)
        # encode cells
        cell_embed = self.token_embedding(cells_split).transpose(0,1).contiguous()
        cell_out, cell_hidden = self.table_encoder(self.cell_lstm, cell_embed, cells_split_len, cells_split_marker, hidden=None, total_length=self.args.cells_token_max_len)
        if self.args.attn_concat:
            col_contex, col_align_score = self.col_pointer_network(input=token_out.transpose(0, 1).contiguous(),
                                                       context=col_out.transpose(0, 1).contiguous(),
                                                       context_lengths=columns_split_marker_len - 1,
                                                       context_max_len=self.args.columns_split_marker_max_len - 1)
            cell_contex, cell_align_score = self.cell_pointer_network(input=token_out.transpose(0, 1).contiguous(),
                                                                      context=cell_out.transpose(0, 1).contiguous(),
                                                                      context_lengths=cells_split_marker_len - 1,
                                                                      context_max_len=self.args.cells_split_marker_max_len - 1)
            col_contex, cell_contex = col_contex.transpose(0, 1).contiguous(), cell_contex.transpose(0, 1).contiguous()
        else:
            # concat token_out and hidden, todo: more layers -> modify fix_hidden
            col_contex, cell_contex = fix_hidden(col_hidden[0]).expand(self.args.tokenize_max_len, batch_size, 2 * self.args.hidden_size), fix_hidden(cell_hidden[0]).expand(self.args.tokenize_max_len, batch_size, 2 * self.args.hidden_size)
        if self.args.cell_info:
            # (tokenize_max_len, batch_size, 6 * hidden_size)
            gate_input = torch.cat([token_out, col_contex, cell_contex], -1)
        else:
            gate_input = torch.cat([token_out, col_contex], -1)
        # (batch_size, tokenize_max_len, 3)
        gate_output = self.gate(gate_input).transpose(0, 1).contiguous()
        if self.args.attn_concat:
            pass
        else:
            # pointer_network; _, (batch_size, tokenize_max_len, columns_split_marker_max_len - 1)
            col_contex, col_align_score = self.col_pointer_network(input=token_out.transpose(0, 1).contiguous(), context=col_out.transpose(0, 1).contiguous(), context_lengths=columns_split_marker_len - 1, context_max_len=self.args.columns_split_marker_max_len - 1)
        # gate_col; (batch_size, tokenize_max_len, columns_split_marker_max_len - 1)
        gate_col = gate_output[:, :, 1].unsqueeze(-1).expand(col_align_score.size()) * col_align_score
        pointer_align_scores = torch.cat([gate_output[:, :, 0].unsqueeze(-1), gate_col, gate_output[:, :, 2].unsqueeze(-1)], dim=-1)
        logger.debug('pointer_align_scores')
        logger.debug(pointer_align_scores)
        # _, _, (batch_size, tgt_len, src_len or class_num)
        return gate_output, col_align_score, pointer_align_scores

    def forward_loss(self, inputs, labels):
        gate_output, _, pointer_align_scores = self.forward(inputs)
        tokenize_len = inputs[0][1]
        mask = sequence_mask(tokenize_len, max_len=self.args.tokenize_max_len).to(self.args.device)
        loss = -self.crf(pointer_align_scores, labels, mask=mask)
        loss /= labels.size(1)
        return loss
