# coding: utf-8

import sys
import torch
import random
import logging
import numpy as np
from torch import nn
from utils import runBiRNN, sequence_mask
from models.modules.TableRNNEncoder import TableRNNEncoder
from models.modules.GlobalAttention import GlobalAttention
from sklearn.metrics import confusion_matrix
from models.modules.PointerNet import PointerNetRNNDecoder
from models.modules.PointerNetDecoderStep import Decoder
from models.modules.Decoder import AttnDecoderRNN

logger = logging.getLogger('binding')
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile='full')


class Baseline(nn.Module):
    def __init__(self, args):
        super(Baseline, self).__init__()
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
        # table_encoder
        self.table_bilstm = nn.LSTM(args.word_dim, args.hidden_size, bidirectional=True, batch_first=False,
                               num_layers=args.num_layers, dropout=args.dropout_p)
        self.table_encoder = TableRNNEncoder(self.args)
        # question-table attention
        self.ques_table_attn = GlobalAttention(self.args, dim=2 * self.hidden_size, attn_type="mlp")
        # top_encoder_lstm
        self.top_encoder_lstm = nn.LSTM(2 * args.hidden_size, args.hidden_size, bidirectional=True, batch_first=False,
                                  num_layers=args.num_layers, dropout=args.dropout_p)
        # point_net_decoder
        # self.pointer_net_decoder = Decoder(embedding_dim=2*self.args.hidden_size, hidden_dim=2*self.args.hidden_size, cuda=self.args.cuda)
        self.pointer_net_decoder = PointerNetRNNDecoder(self.args, input_dim=self.args.word_dim)
        self.attn_decoder = AttnDecoderRNN(self.args)
        # decoder_input
        self.decoder_input = nn.Parameter(torch.zeros(1, 2 * self.args.hidden_size), requires_grad=False).to(self.args.device)
        # unk tensor
        self.unk_tensor = nn.Parameter(torch.zeros(1, self.args.word_dim), requires_grad=False).to(self.args.device)
        self.init_parameters()
        self.gate = nn.Linear(2 * self.args.hidden_size, 2)
        self.transform_in = nn.Linear(4 * self.args.hidden_size, 2 * self.args.hidden_size)

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.decoder_input)
        torch.nn.init.xavier_uniform_(self.unk_tensor)

    def forward(self, inputs):
        # unpack inputs to data
        tokenize, tokenize_len = inputs[0]  # _, (batch_size)
        pos_tag = inputs[1][0]
        columns_split, columns_split_len = inputs[2]
        columns_split_marker, columns_split_marker_len = inputs[3]  # _, (batch_size)
        # cells_split, cells_split_len = inputs[4]
        # cells_split_marker, cells_split_marker_len = inputs[5]  # _, (batch_size)
        batch_size = tokenize.size(0)
        # encode token
        token_embed = self.token_embedding(tokenize)
        token_embed = token_embed.transpose(0, 1).contiguous()  # (tokenize_max_len, batch_size, word_dim)
        logger.debug('token_embed')
        logger.debug(token_embed)
        # add pos_tag on token_embed
        pos_tag_embed = self.pos_tag_embedding(pos_tag).transpose(0, 1).contiguous()  # (tokenize_max_len, batch_size, word_dim)
        token_embed += pos_tag_embed
        # unk_tensor
        # unk_tensor = self.unk_tensor.unsqueeze(0).expand(batch_size, 1, -1).transpose(0, 1).contiguous()  # (1, batch_size, word_dim)
        # input_tensor = torch.cat([unk_tensor, token_embed], dim=0)  # (tokenize_max_len + 1, batch_size, word_dim)
        # run token lstm
        token_out, token_hidden = runBiRNN(self.token_lstm, token_embed, tokenize_len, total_length=self.args.tokenize_max_len)  # (tokenize_max_len, batch_size, 2*hidden_size), _
        # encode table
        col_embed = self.token_embedding(columns_split).transpose(0, 1).contiguous()  # (columns_token_max_len, batch_size, word_dim)
        col_out, col_hidden = self.table_encoder(self.token_lstm, col_embed, columns_split_len, columns_split_marker, hidden=token_hidden, total_length=self.args.columns_token_max_len)  # (columns_split_marker_max_len - 1, batch_size, 2*hidden_size)
        memory_bank = torch.cat([token_out, col_out], dim=0).transpose(0, 1).contiguous()

        # decode one step
        pointer_align_scores, _, _ = self.pointer_net_decoder(tgt=token_embed, src=memory_bank, hidden=col_hidden,
                                                                tgt_lengths=tokenize_len,
                                                                tgt_max_len=self.args.tokenize_max_len,
                                                                src_lengths=None,
                                                                src_max_len=None)
        return pointer_align_scores

        # # decode step by step
        # decoder_input1 = self.decoder_input.unsqueeze(0).expand(batch_size, 1, -1).transpose(0, 1).contiguous()  # (1, batch_size, 2 * self.args.hidden_size)
        # decoder_input2 = token_out[0].unsqueeze(0)
        # decoder_input = self.transform_in(torch.cat([decoder_input1, decoder_input2], dim=-1))
        # pointer_align_scores = torch.zeros(self.args.tokenize_max_len, batch_size, self.args.tokenize_max_len + self.args.columns_split_marker_max_len - 1).to(self.args.device)
        # gate_scores = torch.zeros(self.args.tokenize_max_len, batch_size, self.args.tokenize_max_len + self.args.columns_split_marker_max_len - 1).to(self.args.device)
        # # use_teacher_forcing = True if random.random() < self.args.teacher_forcing_ratio else False
        # for ti in range(self.args.tokenize_max_len):
        #     pointer_align_score, output, hidden = self.pointer_net_decoder(tgt=decoder_input, src=memory_bank, hidden=col_hidden,
        #                                                            src_lengths=None,
        #                                                            src_max_len=None)  # (batch_size, 1, tokenize_max_len + columns_split_marker_max_len), _
        #     self.gate(output)
        #     # (batch_size, 1)
        #     pointer1 = torch.max(pointer_align_score, -1)[1]
        #     pointer2 = torch.LongTensor([ti + 1]).unsqueeze(0).expand(batch_size, 1).to(self.args.device)
        #     # if use_teacher_forcing:
        #     #     pointer = label[:, ti].unsqueeze(1)
        #     batch_index = torch.LongTensor(range(batch_size)).contiguous().view(batch_size, 1)
        #     decoder_input1 = memory_bank[batch_index, pointer1, :].transpose(0, 1).contiguous().detach()
        #     decoder_input2 = memory_bank[batch_index, pointer2, :].transpose(0, 1).contiguous().detach()
        #     decoder_input = self.transform_in(torch.cat([decoder_input1, decoder_input2], dim=-1))
        #     pointer_align_scores[ti] = pointer_align_score.transpose(0, 1).contiguous()
        # 
        # # (batch_size, tgt_len, src_len)
        # return pointer_align_scores.transpose(0, 1).contiguous()
