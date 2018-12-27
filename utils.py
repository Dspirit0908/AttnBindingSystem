# coding: utf-8

import time
import json
import nltk
import torch
import random
import functools
import numpy as np
from torch import nn
from gensim.models import KeyedVectors
from functools import reduce
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from stanza.nlp.corenlp import CoreNLPClient
from config import wikisql_path, preprocess_path, word_embedding_path

client = None
UNK_WORD = '<unk>'
SPLIT_WORD = '<|>'
PAD_WORD = '<blank>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
special_token_list = [UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD, SPLIT_WORD]
special_token_vocab = dict(list(zip(special_token_list, list(range(len(special_token_list))))))


def read_json(path, key):
    all_infos = {}
    with open(path) as f:
        for line in f:
            info = json.loads(line.strip())
            all_infos[info[key]] = info
    return all_infos


def get_wikisql_path(mode):
    return wikisql_path + mode + '.jsonl'


def get_wikisql_tables_path(mode):
    return wikisql_path + mode + '.tables.jsonl'


def get_preprocess_path(mode):
    return preprocess_path + mode + '.jsonl'


def get_annotate(sentence, lower=True):
    # notice return 4 infos
    # todo: handle [Salmonella spp.] -> ['salmonella', 'spp.', '.']
    global client
    if client is None:
        client = CoreNLPClient(server='http://localhost:9000', default_annotators=['ssplit', 'tokenize', 'pos'])
    tokenize, origin, pos_tag, after = [], [], [], []
    for s in client.annotate(sentence):
        for t in s:
            if lower:
                tokenize.append(t.word.lower()), origin.append(t.originalText.lower()), pos_tag.append(t.pos), after.append(t.after)
            else:
                tokenize.append(t.word), origin.append(t.originalText), pos_tag.append(t.pos), after.append(t.after)
    return tokenize, origin, pos_tag, after


def get_ngram(s_list):
    ngram = set()
    for length in range(1, len(s_list) + 1):
        for i in range(len(s_list) - length + 1):
            ngram.add(''.join(s_list[i:i + length]).lower())
    return ngram


def get_split(iter, lower):
    """
    get split and split_marker.
    """
    if len(iter) == 0: return [], 0, [], 0
    # if lower is False: [['Player'], ['No', '.'], ['Nationality'], ['Position'], ['Years', 'in', 'Toronto'], ['School', '/', 'Club', 'Team']]
    columns = list(map(lambda column: get_annotate(column, lower)[0], iter))
    # ['<|>', 'Player', '<|>', 'No', '.', '<|>', 'Nationality', '<|>', 'Position', '<|>', 'Years', 'in', 'Toronto', '<|>', 'School', '/', 'Club', 'Team', '<|>']
    columns_split = [SPLIT_WORD] + list(reduce(lambda x, y: x + [SPLIT_WORD] + y, columns)) + [SPLIT_WORD]
    columns_split_len = len(columns_split)
    # [0, 2, 5, 7, 9, 13, 18]
    columns_split_marker = [index for index in range(len(columns_split)) if columns_split[index] == SPLIT_WORD]
    columns_split_marker_len = len(columns_split_marker)
    return columns_split, columns_split_len, columns_split_marker, columns_split_marker_len


def preprocess(mode, lower=True):
    """
    only need at beginning. stanza may crash on Windows, can work on Linux.
    :param mode: 'train', 'dev' or 'test'.
    """
    print('preprocessing {}'.format(mode))
    # load label info and table info
    label_path = './data/' + 'sql_label_' + mode + '_final_label1.json'
    label_info = read_json(label_path, 'Utterance')
    table_info = read_json(get_wikisql_tables_path(mode), 'id')
    # preprocess data
    data_path, out_path = get_wikisql_path(mode), get_preprocess_path(mode)
    with open(data_path) as f, open(out_path, 'w') as out_f:
        UNK_TERM = {'CoreTerm', 'UnknownTerm', 'AdjectiveTerm', 'VisualTerm'}
        for line in f:
            info = json.loads(line.strip())
            info['tokenize'], info['original'], info['pos_tag'], info['after'] = get_annotate(info['question'], lower=lower)
            assert len(info['tokenize']) == len(info['original']) == len(info['pos_tag']) == len(info['after'])
            # get cells
            cells = set()
            for s_list in table_info[info['table_id']]['rows']:
                for word in s_list:
                    cells.add(str(word))
            # filter cells
            # the first way: need handle "cells": ["1", "8", "8abx15", "5"]
            # cells = [cell.lower() for cell in cells if cell.lower() in info['question'].lower()]
            # the second way
            # todo: (1980.0 in question, 1980 in cell; 14 in question, +14/14. in cell)
            tokenize_ngram = get_ngram(info['original'])
            # must to lower() for match value in cells and question
            info['cells'] = [cell.lower() for cell in cells if cell.lower().replace(' ', '') in tokenize_ngram]
            # try get label
            info['label'] = []
            if info['question'] in label_info:
                the_label_info = label_info[info['question']]['Idx2label']
                the_label_info = sorted(the_label_info.items(), key=lambda x: int(x[0]))
                # ["CoreTerm_who_SpellCorrectedString, ExactMatch, _BE", ...]
                the_label_info = [list(x[1][0].keys())[0] for x in the_label_info]
                # need check word tokenize
                if len(info['tokenize']) == len(the_label_info):
                    try:
                        for index, label in enumerate(the_label_info):
                            label_split = label.split('_')
                            if label_split[0] in UNK_TERM:
                                info['label'].append(UNK_WORD)
                            elif label_split[0] == 'ValueTerm':
                                value = '_'.join(label_split[1:-2]).lower()
                                info['label'].append('Value_' + str(info['cells'].index(value)))
                            # todo: NumberRangeTerm need to handle some special situations, use SQL?
                            elif label_split[0] == 'NumberRangeTerm':
                                try:
                                    value = '_'.join(label_split[1:-2]).lower()
                                    info['label'].append('Value_' + str(info['cells'].index(value)))
                                except Exception as e:
                                    value = '_'.join(label_split[1:-2]).lower()
                                    if 'than' in value:
                                        value = value.split(' than ')[1]
                                        if value.endswith('.0'):
                                            value = value[:-2]
                                        info['label'].append('Value_' + str(info['cells'].index(value)))
                                    else:
                                        raise Exception("Value Not Handle", value)
                            elif label_split[0] == 'ColumnTerm':
                                column = '_'.join(label_split[1:-2])
                                info['label'].append('Column_' + str(table_info[info['table_id']]['header'].index(column)))
                            else:
                                raise Exception("Label Not Handle", label)
                        assert len(info['label']) == len(info['tokenize'])
                    except Exception as e:
                        print(e)
                        print(info['question'])
                        info['label'] = []
            # get columns/cells split and split_marker
            info['columns_split'], info['columns_split_len'], info['columns_split_marker'], info['columns_split_marker_len'] = get_split(table_info[info['table_id']]['header'], lower=lower)
            info['cells_split'], info['cells_split_len'], info['cells_split_marker'], info['cells_split_marker_len'] = get_split(info['cells'], lower=lower)
            out_f.write(json.dumps(info) + '\n')


def load_data(path, vocab=False, only_label=False):
    print('loading {}'.format(path))
    tokenize_list, tokenize_len_list = [], []
    pos_tag_list = []
    table_id_list = []
    columns_split_list, columns_split_len_list, columns_split_marker_list, columns_split_marker_len_list = [], [], [], []
    cells_split_list, cells_split_len_list, cells_split_marker_list, cells_split_marker_len_list = [], [], [], []
    label_list = []
    # a list, list of list, list of list of list
    sql_sel_col_list, sql_conds_cols_list, sql_conds_values_list = [], [], []
    with open(path) as f:
        for line in f:
            info = json.loads(line.strip())
            # get label
            label = info['label']
            # if only_label, pass when the label is []. need check only_label at first.
            if only_label:
                if len(label) == 0:
                    continue
            # get tokenize
            tokenize = info['tokenize']
            # get conds
            conds_cols, conds_values = [], []
            conds_values_flag = True
            for cond in info['sql']['conds']:
                conds_cols.append(cond[0])
                value_list = get_annotate(str(cond[2]), lower=True)[0]
                value_index = find_value_index(value_list, token_list=tokenize)
                # todo: handle this situation
                if value_index is None:
                    conds_values_flag = False
                    print(value_list[0]), print(tokenize)
                    break
                else:
                    conds_values.append(value_index)
            # append
            if conds_values_flag:
                tokenize_list.append(tokenize), tokenize_len_list.append(len(tokenize))
                pos_tag_list.append(info['pos_tag'])
                table_id_list.append(info['table_id'])
                columns_split_list.append(info['columns_split']), columns_split_len_list.append(info['columns_split_len'])
                columns_split_marker_list.append(info['columns_split_marker']), columns_split_marker_len_list.append(info['columns_split_marker_len'])
                cells_split_list.append(info['cells_split']), cells_split_len_list.append(info['cells_split_len'])
                cells_split_marker_list.append(info['cells_split_marker']), cells_split_marker_len_list.append(info['cells_split_marker_len'])
                label_list.append(label)
                sql_sel_col_list.append(info['sql']['sel']), sql_conds_cols_list.append(conds_cols), sql_conds_values_list.append(conds_values)
    if vocab:
        return tokenize_list, columns_split_list
    else:
        # check
        assert len(tokenize_list) == len(tokenize_len_list) == len(pos_tag_list) == len(table_id_list)\
               == len(columns_split_list) == len(cells_split_list) == len(label_list) == len(sql_sel_col_list)\
               == len(sql_conds_cols_list) == len(sql_conds_values_list)
        return tokenize_list, tokenize_len_list, pos_tag_list, table_id_list,\
               (columns_split_list, columns_split_len_list, columns_split_marker_list, columns_split_marker_len_list),\
               (cells_split_list, cells_split_len_list, cells_split_marker_list, cells_split_marker_len_list),\
               label_list, sql_sel_col_list, sql_conds_cols_list, sql_conds_values_list


def load_word_embedding(word_dim, vocab, max_vocab_size=None):
    embedding_model = KeyedVectors.load_word2vec_format(word_embedding_path)
    vocab_size = len(vocab) if max_vocab_size is None else max_vocab_size
    embed_matrix = np.random.uniform(-0.25, 0.25, size=(vocab_size, word_dim))
    for word, i in vocab.items():
        if i >= vocab_size:
            continue
        if word in embedding_model:
            embedding_vector = embedding_model[word]
            if embedding_vector is not None:
                embed_matrix[i] = embedding_vector
        else:
            pass
            # w_count = 0
            # for w in word.split(' '):
            #     if w in embedding_model:
            #         w_count += 1
            #         embed_matrix[i] += embedding_model[w]
            # if w_count != 0:
            #     embed_matrix[i] /= w_count
    return embed_matrix


def build_vocab(m_lists, pre_func=None, init_vocab=None, sort=True, min_word_freq=1):
    """
    :param m_lists: short for many lists, means list of list.
    :param pre_func: preprocess function for every word in a single list.
    :param init_vocab: init_vocab.
    :param min_count: min_count.
    :return: word2index and index2word.
    """
    # get word count
    word_count = {}
    for s_list in m_lists:
        for word in s_list:
            if pre_func is not None:
                word = pre_func(word)
            word_count[word] = word_count.get(word, 0) + 1
    # filter rare words
    new_word_count_keys = [key for key in word_count if word_count[key] >= min_word_freq]
    # sort
    if sort:
        new_word_count_keys = sorted(new_word_count_keys, key=lambda x: word_count[x], reverse=True)
    # init
    index2word = {}
    if init_vocab is None:
        word2index = {}
        num = 0
    else:
        word2index = init_vocab
        num = len(init_vocab)
        for k, v in word2index.items():
            index2word[v] = k
    # get word2index and index2word
    word2index.update(dict(list(zip(new_word_count_keys, list(range(num, num + len(new_word_count_keys)))))))
    index2word.update(dict(list(zip(list(range(num, num + len(new_word_count_keys))), new_word_count_keys))))
    return word2index, index2word


def build_all_vocab(init_vocab=None, min_word_freq=1):
    # need to know all the words to filter the pretrained word embeddings
    load_vocab = functools.partial(load_data, vocab=True, only_label=False)
    mode_list = ['train', 'dev', 'test']
    vocab = []
    for mode in mode_list:
        tokenize_list, columns_split_list = load_vocab(get_preprocess_path(mode))
        vocab.extend(tokenize_list)
        vocab.extend(columns_split_list)
    word2index, index2word = build_vocab(vocab, init_vocab=init_vocab, min_word_freq=min_word_freq)
    return word2index, index2word


def change2idx(m_lists, vocab, oov_token=0, name='change2idx'):
    idxs_list = []
    oov_count, total = 0, 0
    for s_list in m_lists:
        # change2idx
        idxs = []
        for word in s_list:
            if word not in vocab:
                oov_count += 1
            total += 1
            idxs.append(vocab.get(word, oov_token))
        idxs_list.append(idxs)
    print('{}: oov_count - {}, total_count - {}'.format(name, oov_count, total))
    return idxs_list


def pad(m_lists, max_len, pad_token=0):
    idxs_list = []
    for s_list in m_lists:
        if len(s_list) < max_len:
            pad = [pad_token for _ in range(max_len - len(s_list))]
            s_list.extend(pad)
        else:
            s_list = s_list[:max_len]
        idxs_list.append(s_list)
    return idxs_list


def find_value_index(value_list, token_list):
    """
    :param value_list: value list
    :param token_list: token list
    :return: a list: [start_index, end_index + 1] or None
    """
    value_length = len(value_list)
    for start_index in range(len(token_list) - value_length + 1):
        if value_list[:] == token_list[start_index:start_index + value_length]:
            return [start_index, start_index + value_length]
    return None


def max_len_of_m_lists(m_lists):
    max_len = 0
    for s_list in m_lists:
        max_len = max(max_len, len(s_list))
    return max_len


# def merge_dicts(dicts, copy=True):
#     """
#     :param dicts: a list of dict.
#     :param copy: copy dicts[0] for merge or use dicts[0] directly.
#     :return: a merged dict.
#     """
#     if copy:
#         dict_0 = dicts[0].copy()
#     else:
#         dict_0 = dicts[0]
#     max_value = max(dict_0.values())
#     for index in range(1, len(dicts)):
#         single_dict = dicts[index]
#         for key, value in single_dict:
#             if key in dict_0:
#                 pass
#             else:
#                 max_value += 1
#                 dict_0[key] = max_value
#     return dict_0


def set_seed(seed):
    """Set random seed everywhere."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def runBiRNN(rnn, inputs, seq_lengths, hidden=None, total_length=None):
    """
    :param rnn: RNN instance
    :param inputs: FloatTensor, shape [batch, time, dim] if rnn.batch_first else [time, batch, dim]
    :param seq_lengths: LongTensor shape [batch]
    :return: the result of rnn layer,
    """
    batch_first = rnn.batch_first
    # assume seq_lengths = [3, 5, 2]
    # 对序列长度进行排序(降序), sorted_seq_lengths = [5, 3, 2]
    # indices 为 [1, 0, 2], indices 的值可以这么用语言表述
    # 原来 batch 中在 0 位置的值, 现在在位置 1 上.
    # 原来 batch 中在 1 位置的值, 现在在位置 0 上.
    # 原来 batch 中在 2 位置的值, 现在在位置 2 上.
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)

    # 如果我们想要将计算的结果恢复排序前的顺序的话,
    # 只需要对 indices 再次排序(升序),会得到 [0, 1, 2],
    # desorted_indices 的结果就是 [1, 0, 2]
    # 使用 desorted_indices 对计算结果进行索引就可以了.
    _, desorted_indices = torch.sort(indices, descending=False)

    # 对原始序列进行排序
    if batch_first:
        inputs = inputs[indices]
    else:
        inputs = inputs[:, indices]
    packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs,
                                                      sorted_seq_lengths.cpu().numpy(),
                                                      batch_first=batch_first)

    res, hidden = rnn(packed_inputs, hidden)

    padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=batch_first, total_length=total_length)
    # 恢复排序前的样本顺序
    if batch_first:
        desorted_res = padded_res[desorted_indices]
    else:
        desorted_res = padded_res[:, desorted_indices]
    
    if isinstance(hidden, tuple):
        hidden = list(hidden)
        hidden[0] = hidden[0][:, desorted_indices]
        hidden[1] = hidden[1][:, desorted_indices]
    else:
        hidden = hidden[:, desorted_indices]

    return desorted_res, hidden


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def count_of_diff(l1, l2):
    assert len(l1) == len(l2)
    count_of_diff, index = 0, 0
    wrong_indexs = []
    for x, y in zip(l1, l2):
        if x != y:
            count_of_diff += 1
            wrong_indexs.append(index)
        index += 1
    return count_of_diff, wrong_indexs


def compare_sql():
    pass


if __name__ == '__main__':
    # preprocess all
    mode_list = ['train', 'dev', 'test']
    table_id_set_list = []
    for index, mode in enumerate(mode_list):
        preprocess(mode)
    pass
