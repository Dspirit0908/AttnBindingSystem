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
from pytorch_pretrained_bert import BertModel

logger = logging.getLogger('binding')
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile='full')


class BertGate(nn.Module):
    def __init__(self, args):
        super(BertGate, self).__init__()
        # args
        self.args = args
        self.bert_model = BertModel.from_pretrained(self.args.bert_model)
        # # pos_tag embedding
        # self.pos_tag_embedding = nn.Embedding(len(args.pos_tag_vocab), args.word_dim)
        # gate
        self.gate = nn.Linear(2 * self.bert_model.config.hidden_size, self.args.gate_class)
        # column pointer network
        self.column_pointer_network = GlobalAttention(args=self.args, dim=self.bert_model.config.hidden_size, is_transform_out=False, attn_type="mlp")
        if self.args.crf:
            # todo: set num for baseline
            if self.args.model == 'gate':
                if self.args.cell_info:
                    self.crf = ConditionalRandomField(1 + self.args.bert_columns_split_marker_max_len - 1 + self.args.bert_cells_split_marker_max_len - 1)
                else:
                    self.crf = ConditionalRandomField(1 + self.args.bert_columns_split_marker_max_len - 1 + 1)
            else:
                raise NotImplementedError

    def forward(self, inputs):
        # unpack inputs to data
        tokenize, tokenize_len, tokenize_marker, tokenize_marker_len = inputs[0]  # (batch_size, tokenize_max_len), (batch_size), (batch_size, tokenize_max_len), (batch_size)
        columns_split, columns_split_len, columns_split_marker, columns_split_marker_len = inputs[1]  # (batch_size, bert_columns_split_max_len), _, _, _
        cells_split, cells_split_len, cells_split_marker, cells_split_marker_len = inputs[2]
        batch_size = tokenize.size(0)
        # encode token and columns
        # (batch_size, tokenize_max_len + bert_columns_split_max_len)
        bert_tokens_and_cols = torch.cat([tokenize, columns_split], dim=-1)
        bert_tokens_segments, bert_columns_segments = torch.zeros_like(tokenize), torch.ones_like(columns_split)
        # (batch_size, tokenize_max_len + bert_columns_split_max_len)
        bert_segments = torch.cat([bert_tokens_segments, bert_columns_segments], dim=-1)
        bert_tokens_mask, bert_columns_mask = sequence_mask(tokenize_len, max_len=self.args.bert_tokenize_max_len), sequence_mask(columns_split_len, max_len=self.args.bert_columns_split_max_len)
        bert_mask = torch.cat([bert_tokens_mask, bert_columns_mask], dim=-1)
        # (batch_size, tokenize_max_len + bert_columns_split_max_len, self.bert_model.config.hidden_size), _
        bert_output, _ = self.bert_model(bert_tokens_and_cols, bert_segments, attention_mask=bert_mask, output_all_encoded_layers=False)
        # (batch_size, tokenize_max_len, self.bert_model.config.hidden_size), (batch_size, bert_columns_split_max_len, self.bert_model.config.hidden_size)
        bert_tokens_output, bert_columns_output = bert_output[:, :self.args.bert_tokenize_max_len, :], bert_output[:, self.args.bert_tokenize_max_len:, :]
        # add sub_tokens
        bert_tokens_output_cumsum, bert_columns_output_cumsum = torch.cumsum(bert_tokens_output, dim=1), torch.cumsum(bert_columns_output, dim=1)
        batch_index = torch.LongTensor(range(batch_size)).unsqueeze(-1).to(self.args.device)
        # (batch_size, tokenize_max_len - 1), (batch_size, bert_columns_split_max_len - 1)
        tokens_batch_index, columns_batch_index = batch_index.expand(tokenize_marker.size(0), tokenize_marker.size(1) - 1), batch_index.expand(columns_split_marker.size(0), columns_split_marker.size(1) - 1)
        # add sub_tokens for tokens
        right_tokens_marker, left_tokens_marker = tokenize_marker[:, 1:], tokenize_marker[:, :-1]
        # (batch_size, tokenize_max_len - 1, self.bert_model.config.hidden_size), (batch_size, tokenize_max_len - 1, self.bert_model.config.hidden_size)
        right_bert_tokens_output_cumsum, left_bert_tokens_output_cumsum = bert_tokens_output_cumsum[tokens_batch_index, right_tokens_marker, :], bert_tokens_output_cumsum[tokens_batch_index, left_tokens_marker, :]
        # (batch_size, tokenize_max_len, self.bert_model.config.hidden_size)
        bert_tokens_sum = torch.cat([bert_tokens_output_cumsum[batch_index, tokenize_marker[:, 0].unsqueeze(1), :], right_bert_tokens_output_cumsum - left_bert_tokens_output_cumsum], dim=1)
        # add sub_tokens for columns
        right_columns_split_marker, left_columns_split_marker = (columns_split_marker - 1)[:, 1:], columns_split_marker[:, :-1]
        right_bert_columns_output_cumsum, left_bert_columns_output_cumsum = bert_columns_output_cumsum[columns_batch_index, right_columns_split_marker, :], bert_columns_output_cumsum[columns_batch_index, left_columns_split_marker, :]
        # (batch_size, bert_columns_split_max_len - 1, self.bert_model.config.hidden_size)
        bert_columns_split_sum = right_bert_columns_output_cumsum - left_bert_columns_output_cumsum
        if self.args.attn_concat:
            # (batch_size, tokenize_max_len, self.bert_model.config.hidden_size), _
            column_attn_h, column_align_score = self.column_pointer_network(input=bert_tokens_sum, context=bert_columns_split_sum,
                                                                            context_lengths=columns_split_marker_len - 1, context_max_len=self.args.bert_columns_split_marker_max_len - 1)
        else:
            raise NotImplementedError
        # gate_input = torch.cat([column_attn_h, bert_tokens_sum], dim=-1)
        gate_input = column_attn_h
        # (batch_size, tokenize_max_len, self.args.gate_class)
        gate_output = self.gate(gate_input)
        gate_output_column = gate_output[:, :, 1].unsqueeze(-1).expand(column_align_score.size()) * column_align_score
        pointer_align_scores = torch.cat([gate_output[:, :, 0].unsqueeze(-1), gate_output_column, gate_output[:, :, 2].unsqueeze(-1)], dim=-1)
        return gate_output, column_align_score, pointer_align_scores

    def forward_loss(self, inputs, labels):
        gate_output, _, pointer_align_scores = self.forward(inputs)
        tokenize_len = inputs[0][1]
        mask = sequence_mask(tokenize_len, max_len=self.args.bert_tokenize_max_len).to(self.args.device)
        loss = -self.crf(pointer_align_scores, labels, mask=mask)
        loss /= labels.size(1)
        return loss
