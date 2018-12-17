# coding: utf-8

import sys
import torch
import numpy as np
from torch import nn
from torchcrf import CRF
from utils import runBiRNN, sequence_mask
from models.modules.TableRNNEncoder import TableRNNEncoder
from models.modules.Attention import Attention
from sklearn.metrics import confusion_matrix
from models.modules.PointerNet import PointerNetRNNDecoder
from models.modules.PointerNetDecoderStep import Decoder
from models.modules.Decoder import AttnDecoderRNN
np.set_printoptions(threshold=np.inf)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # args
        self.args = args
        self.hidden_size = args.hidden_size
        # embedding
        self.token_embedding = nn.Embedding(args.vocab_size, args.word_dim)
        if args.embed_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(args.embed_matrix))
        # token_lstm
        self.token_lstm = nn.LSTM(args.word_dim, args.hidden_size, bidirectional=True, batch_first=False,
                               num_layers=args.num_layers, dropout=args.dropout_p)
        # table_encoder
        self.table_bilstm = nn.LSTM(args.word_dim, args.hidden_size, bidirectional=True, batch_first=False,
                               num_layers=args.num_layers, dropout=args.dropout_p)
        self.table_encoder = TableRNNEncoder(self.table_bilstm, self.args)
        # question-table attention
        self.ques_table_attn = Attention('general', dim=2*self.hidden_size, args=self.args)
        # top_encoder_lstm
        self.top_encoder_lstm = nn.LSTM(2 * args.hidden_size, args.hidden_size, bidirectional=True, batch_first=False,
                                  num_layers=args.num_layers, dropout=args.dropout_p)
        # point_net_decoder
        # self.pointer_net_decoder = Decoder(embedding_dim=2*self.args.hidden_size, hidden_dim=2*self.args.hidden_size, cuda=self.args.cuda)
        self.pointer_net_decoder = PointerNetRNNDecoder(self.args)
        self.attn_decoder = AttnDecoderRNN(self.args)
        self.decoder_input = nn.Parameter(torch.ones(1, 2*self.args.hidden_size), requires_grad=False)
        torch.nn.init.xavier_uniform(self.decoder_input)
        # unk tensor
        self.unk_tensor = nn.Parameter(torch.rand(self.args.batch_size, 1, 2*self.args.hidden_size))

    def forward(self, inputs):
        # unpack inputs to data
        tokenize, tokenize_len = inputs[0]  # _, (batch_size)
        pos_tag = inputs[1][0]
        columns_split, columns_split_len = inputs[2]
        columns_split_marker, columns_split_marker_len = inputs[3]  # _, (batch_size)
        batch_size = tokenize.size(0)
        # encode token
        token_embed = self.token_embedding(tokenize)
        token_embed = token_embed.contiguous().transpose(0, 1).contiguous()  # (tokenize_max_len, batch_size, word_dim)
        token_out, token_hidden = runBiRNN(self.token_lstm, token_embed, tokenize_len, total_length=self.args.tokenize_max_len)  # (tokenize_max_len, batch_size, 2*hidden_size), _
        # encode table
        table_embed = self.token_embedding(columns_split).contiguous().transpose(0, 1).contiguous()  # (column_token_max_len, batch_size, word_dim)
        table_out = self.table_encoder(table_embed, columns_split_len, columns_split_marker)  # (columns_split_marker_max_len - 1, batch_size, 2*hidden_size)
        # question-table attention
        # table_out: not zero vector for pad in the last
        token_out, table_out = token_out.transpose(0, 1), table_out.transpose(0, 1)
        # attn_h: not zero vector for pad in the last, align_score: not zero vector for pad in the last, zero vector for pad in the right
        attn_h, align_score = self.ques_table_attn(table_out, token_out, columns_split_marker_len - 1, src_max_len=self.args.columns_split_marker_max_len - 1)  # (batch_size, tokenize_max_len, 2*hidden_size), (batch_size, tokenize_max_len, columns_split_marker_max_len - 1)
        # top_encoder_lstm
        attn_h = attn_h.transpose(0, 1)  # (tokenize_max_len, batch_size, 2*hidden_size)
        top_out, top_hidden = runBiRNN(self.top_encoder_lstm, attn_h, tokenize_len, total_length=self.args.tokenize_max_len)  # (tokenize_max_len, batch_size, 2*hidden_size), _
        # point_net_decoder
        top_out = top_out.transpose(0, 1)
        # print(table_out)
        memory_bank = torch.cat([token_out, table_out], dim=1)  # (batch_size, tokenize_max_len + columns_split_marker_max_len - 1, 2*hidden_size)
        decoder_input = self.decoder_input.unsqueeze(1).expand(1, batch_size, -1)
        hidden = top_hidden
        pointer_align_scores = torch.randn(self.args.tokenize_max_len, batch_size, self.args.tokenize_max_len + self.args.columns_split_marker_max_len - 1).cuda()
        for ti in range(self.args.tokenize_max_len):
            pointer_align_score, hidden = self.pointer_net_decoder(memory_bank, decoder_input, hidden,
                                                           src_lengths=self.args.tokenize_max_len + columns_split_marker_len - 1,
                                                           src_max_len=self.args.tokenize_max_len + self.args.columns_split_marker_max_len - 1)  # (batch_size, tokenize_max_len, tokenize_max_len + columns_split_marker_max_len - 1)
            index = torch.max(pointer_align_score, 2)[1]
            batch_index = torch.LongTensor(range(batch_size)).contiguous().view(batch_size, 1)
            decoder_input = memory_bank[batch_index, index, :].transpose(0, 1).contiguous()
            pointer_align_scores[ti] = pointer_align_score.transpose(0, 1).contiguous()

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
        return pointer_align_scores.transpose(0, 1).contiguous()
