# coding: utf-8

import sys
import torch
import random
import logging
import numpy as np
from torch import nn
from utils import runBiRNN, sequence_mask
from models.modules.TableRNNEncoder import TableRNNEncoder
from models.modules.Attention import Attention
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
        self.ques_table_attn = Attention('general', dim=2*self.hidden_size, args=self.args)
        # top_encoder_lstm
        self.top_encoder_lstm = nn.LSTM(2 * args.hidden_size, args.hidden_size, bidirectional=True, batch_first=False,
                                  num_layers=args.num_layers, dropout=args.dropout_p)
        # point_net_decoder
        # self.pointer_net_decoder = Decoder(embedding_dim=2*self.args.hidden_size, hidden_dim=2*self.args.hidden_size, cuda=self.args.cuda)
        self.pointer_net_decoder = PointerNetRNNDecoder(self.args, input_dim=self.args.word_dim)
        self.attn_decoder = AttnDecoderRNN(self.args)
        self.decoder_input = nn.Parameter(torch.zeros(1, 2*self.args.hidden_size), requires_grad=False).to(self.args.device)
        # unk tensor
        self.unk_tensor = nn.Parameter(torch.zeros(1, 2*self.args.hidden_size)).to(self.args.device)
        self.init_parameters()

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
        # run token lstm
        token_out1, token_hidden1 = runBiRNN(self.token_lstm, token_embed, tokenize_len, total_length=self.args.tokenize_max_len)  # (tokenize_max_len, batch_size, 2*hidden_size), _
        token_out, token_hidden = runBiRNN(self.token_lstm, token_embed, tokenize_len, hidden=token_hidden1,
                                           total_length=self.args.tokenize_max_len)  # (tokenize_max_len, batch_size, 2*hidden_size), _
        logger.debug('token_out')
        logger.debug(token_out)
        # encode table
        col_embed = self.token_embedding(columns_split).transpose(0, 1).contiguous()  # (columns_token_max_len, batch_size, word_dim)
        col_out, col_hidden = self.table_encoder(self.token_lstm, col_embed, columns_split_len, columns_split_marker, hidden=token_hidden, total_length=self.args.columns_token_max_len)  # (columns_split_marker_max_len - 1, batch_size, 2*hidden_size)
        # cell_embed = self.token_embedding(cells_split).transpose(0, 1).contiguous()  # (columns_token_max_len, batch_size, word_dim)
        # table_out, table_hidden = self.table_encoder(self.token_lstm, cell_embed, cells_split_len, cells_split_marker, hidden=col_hidden, total_length=self.args.cells_token_max_len)  # (columns_split_marker_max_len - 1, batch_size, 2*hidden_size)
        memory_bank = torch.cat([token_out1, token_out, col_out], dim=0).transpose(0, 1).contiguous()
        unk_tensor = self.unk_tensor.unsqueeze(0).expand(batch_size, 1, -1)
        # memory_bank = torch.cat([memory_bank, unk_tensor], dim=1)
        logger.debug('memory_bank')
        logging.debug(memory_bank)
        # attn_h, align_score = self.ques_table_attn(table_out.transpose(0, 1).contiguous(), token_out.transpose(0, 1).contiguous(), columns_split_marker_len - 1,
        #                                            src_max_len=self.args.columns_split_marker_max_len - 1)
        pointer_align_scores, hidden = self.pointer_net_decoder(memory_bank, token_embed, hidden=col_hidden, tgt_lengths=tokenize_len, tgt_max_len=self.args.tokenize_max_len,
                                                               src_lengths=None,
                                                               src_max_len=None)  # (batch_size, tokenize_max_len, tokenize_max_len + columns_split_marker_max_len), _
        logger.debug('pointer_align_scores')
        logger.debug(pointer_align_scores)
        
        # # question-table attention
        # # table_out: not zero vector for pad in the last
        # token_out, table_out = token_out.transpose(0, 1).contiguous(), table_out.transpose(0, 1).contiguous()
        # # attn_h: not zero vector for pad in the last, align_score: not zero vector for pad in the last, zero vector for pad in the right
        # attn_h, align_score = self.ques_table_attn(table_out, token_out, columns_split_marker_len - 1, src_max_len=self.args.columns_split_marker_max_len - 1)  # (batch_size, tokenize_max_len, 2*hidden_size), (batch_size, tokenize_max_len, columns_split_marker_max_len - 1)
        # # top_encoder_lstm
        # attn_h = attn_h.transpose(0, 1)  # (tokenize_max_len, batch_size, 2*hidden_size)
        # top_out, top_hidden = runBiRNN(self.top_encoder_lstm, attn_h, tokenize_len, total_length=self.args.tokenize_max_len)  # (tokenize_max_len, batch_size, 2*hidden_size), _
        # # point_net_decoder
        # unk_tensor = self.unk_tensor.unsqueeze(0).expand(batch_size, 1, -1)
        # memory_bank = torch.cat([token_out, table_out, unk_tensor], dim=1)  # (batch_size, tokenize_max_len + columns_split_marker_max_len, 2 * hidden_size)
        # # memory_bank = token_out
        # logger.debug('memory_bank')
        # logging.debug(memory_bank)
        # decoder_input, hidden = self.decoder_input.unsqueeze(1).expand(1, batch_size, -1), top_hidden
        # 
        # pointer_align_score, hidden = self.pointer_net_decoder(memory_bank, decoder_input, hidden,
        #                                                            src_lengths=None,
        #                                                            src_max_len=None)  # (batch_size, tokenize_max_len, tokenize_max_len + columns_split_marker_max_len), _
        # 
        # pointer_align_scores = torch.zeros(self.args.tokenize_max_len, batch_size, self.args.tokenize_max_len + self.args.columns_split_marker_max_len).to(self.args.device)
        # use_teacher_forcing = True if random.random() < self.args.teacher_forcing_ratio else False
        # for ti in range(self.args.tokenize_max_len):
        #     pointer_align_score, hidden = self.pointer_net_decoder(memory_bank, decoder_input, hidden,
        #                                                            src_lengths=None,
        #                                                            src_max_len=None)  # (batch_size, tokenize_max_len, tokenize_max_len + columns_split_marker_max_len), _
        #     logger.debug('pointer_align_score')
        #     logging.debug(pointer_align_score.size())
        #     logging.debug(pointer_align_score)
        #     pointer = torch.max(pointer_align_score, 2)[1]
        #     ture_pointer = torch.LongTensor([ti]).unsqueeze(0).expand(batch_size, 1)  # +1?
        #     logger.debug('ture_pointer')
        #     logging.debug(ture_pointer)
        #     logger.debug('pointer')
        #     logging.debug(pointer)
        #     batch_index = torch.LongTensor(range(batch_size)).contiguous().view(batch_size, 1)
        #     if use_teacher_forcing:
        #         pointer = ture_pointer
        #     decoder_input = memory_bank[batch_index, pointer, :].transpose(0, 1).contiguous()
        #     logger.debug('decoder_input')
        #     logging.debug(decoder_input)
        #     pointer_align_scores[ti] = pointer_align_score.transpose(0, 1).contiguous()

        # decoder_input, hidden = self.decoder_input, top_hidden
        # pointer_align_score = []
        # for ti in range(self.args.tokenize_max_len):
        #     align_score, hidden = self.attn_decoder(
        #         decoder_input, hidden, memory_bank, src_lengths=self.args.tokenize_max_len + columns_split_marker_len - 1,
        #         src_max_len=self.args.tokenize_max_len + self.args.columns_split_marker_max_len - 1)
        #     topv, topi = align_score.topk(1)
        #     decoder_input = topi.squeeze(1)
        #     print(decoder_input)
        #     pointer_align_score.append(align_score)
        # def _cat_directions(h):
        #     """ If the encoder is bidirectional, do the following transformation.
        #         (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        #     """
        #     h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        #     return h
        # embed_inputs = torch.cat([token_out, table_out], dim=1)
        # decoder_input0 = self.decoder_input.unsqueeze(0).expand(batch_size, -1)
        # pointer_align_score = self.pointer_net_decoder(embed_inputs, decoder_input0, top_hidden, memory_bank)
        # print(pointer_align_scores.transpose(0, 1).contiguous())
        # logger.debug('pointer_align_scores.transpose(0, 1)')
        # logger.debug(pointer_align_scores.transpose(0, 1).contiguous().size())
        # logger.debug(pointer_align_scores.transpose(0, 1).contiguous())
        return pointer_align_scores
