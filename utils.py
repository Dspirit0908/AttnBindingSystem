# coding: utf-8

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
from config import wikisql_path, preprocess_path, word_embedding_path

UNK_WORD = '<unk>'
PAD_WORD = '<blank>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
SPLIT_WORD = '<|>'
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


def preprocess(mode):
    """
    write [tokenize, pos_tag, lemma] to a new file. only need at beginning.
    # lemma [seems] not good for this task. eg: (ks) -> (k). or need lemma tokens and columns together.
    :param mode: 'train', 'dev' or 'test'.
    :return:
    """
    print('preprocessing {}'.format(mode))
    label_path = './data/' + 'sql_label_' + mode + '_final_label1.json'
    label_info = read_json(label_path, 'Utterance')
    table_info = read_json(get_wikisql_tables_path(mode), 'id')

    data_path, out_path = get_wikisql_path(mode), get_preprocess_path(mode)
    with open(data_path) as f, open(out_path, 'w') as out_f:
        UNK_TERM = {'CoreTerm', 'UnknownTerm', 'AdjectiveTerm', 'VisualTerm'}
        for line in f:
            info = json.loads(line.strip())
            info['tokenize'] = WordPunctTokenizer().tokenize(info['question'])
            info['lemmatize'] = [WordNetLemmatizer().lemmatize(word) for word in info['tokenize']]
            info['pos_tag'] = [x[1] for x in nltk.pos_tag(info['tokenize'])]

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
                                info['label'].append('UNK')
                            elif label_split[0] == 'ValueTerm' or label_split[0] == 'NumberRangeTerm':
                                info['label'].append('Value_' + str(index))
                            elif label_split[0] == 'ColumnTerm':
                                column = '_'.join(label_split[1:-2])
                                info['label'].append('Column_' + str(table_info[info['table_id']]['header'].index(column)))
                            else:
                                print(label)
                        assert len(info['label']) == len(info['tokenize'])
                    except Exception as e:
                        print(e)
                        print(info['question'])
                        info['label'] = []
            assert len(info['tokenize']) == len(info['lemmatize']) == len(info['pos_tag'])
            out_f.write(json.dumps(info) + '\n')


def load_data(path, only_tokenize=False, only_label=False, lower=True):
    print('loading {}'.format(path))
    tokenize_list, tokenize_len_list = [], []
    pos_tag_list = []
    table_id_list = []
    label_list = []
    with open(path) as f:
        for line in f:
            info = json.loads(line.strip())
            if lower:
                tokenize = [word.lower() for word in info['tokenize']]
            else:
                tokenize = info['tokenize']
            pos_tag = info['pos_tag']
            table_id = info['table_id']
            label = info['label']
            # if only_label, pass when the label is []
            if only_label:
                if len(label) == 0:
                    continue
            # append
            tokenize_list.append(tokenize)
            if only_tokenize:
                pass
            else:
                tokenize_len_list.append(len(tokenize))
                pos_tag_list.append(pos_tag)
                table_id_list.append(table_id)
                label_list.append(label)
    if only_tokenize:
        return tokenize_list
    else:
        # check
        assert len(tokenize_list) == len(tokenize_len_list) == len(pos_tag_list) == len(table_id_list) == len(label_list)
        return tokenize_list, tokenize_len_list, pos_tag_list, table_id_list, label_list


def load_tables(path, vocab=False, lower=True):
    print('loading {}'.format(path))
    tables_info = {}
    # for vocab
    columns_split_list = []
    with open(path) as f:
        for line in f:
            info = json.loads(line.strip())
            key = info['id']
            if key not in tables_info:
                tables_info[key] = {}
            if lower:
                columns = list(map(lambda column: WordPunctTokenizer().tokenize(column.lower()), info['header']))
            else:
                # [['Player'], ['No', '.'], ['Nationality'], ['Position'], ['Years', 'in', 'Toronto'], ['School', '/', 'Club', 'Team']]
                columns = list(map(lambda column: WordPunctTokenizer().tokenize(column), info['header']))
            # ['<|>', 'Player', '<|>', 'No', '.', '<|>', 'Nationality', '<|>', 'Position', '<|>', 'Years', 'in', 'Toronto', '<|>', 'School', '/', 'Club', 'Team', '<|>']
            columns_split = [SPLIT_WORD] + list(reduce(lambda x, y: x + [SPLIT_WORD] + y, columns)) + [SPLIT_WORD]
            # [0, 2, 5, 7, 9, 13, 18]
            columns_split_marker = [ index for index in range(len(columns_split)) if columns_split[index] == SPLIT_WORD]
            if vocab:
                columns_split_list.append(columns_split)
            else:
                tables_info[key]['columns_split'] = columns_split
                tables_info[key]['columns_split_marker'] = columns_split_marker
    if vocab:
        return columns_split_list
    else:
        return tables_info


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

    word2index.update(dict(list(zip(new_word_count_keys, list(range(num, num + len(new_word_count_keys)))))))
    index2word.update(dict(list(zip(list(range(num, num + len(new_word_count_keys))), new_word_count_keys))))
    return word2index, index2word


def build_all_vocab():
    # need to know all the words to filter the pretrained word embeddings
    load_only_tokenize = functools.partial(load_data, only_tokenize=True)
    load_columns_vocab = functools.partial(load_tables, vocab=True)
    mode_list = ['train', 'dev', 'test']
    all_tokenize, all_columns = [], []
    for mode in mode_list:
        tokenize = load_only_tokenize(get_preprocess_path(mode))
        all_tokenize.extend(tokenize)
        columns = load_columns_vocab(get_wikisql_tables_path(mode))
        all_columns.extend(columns)
    all_tokenize.extend(all_columns)
    word2index, index2word = build_vocab(all_tokenize)
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


def compare_sql_col():
    pass


if __name__ == '__main__':
    mode_list = ['train', 'dev', 'test']
    table_id_set_list = []
    for index, mode in enumerate(mode_list):
        table_id_set_list.append(set(load_tables(get_wikisql_tables_path(mode)).keys()))
        print(len(table_id_set_list[index]))
    print(len(table_id_set_list[0] - table_id_set_list[1]))
    print(len(table_id_set_list[0] - table_id_set_list[2]))
    print(len(table_id_set_list[1] - table_id_set_list[2]))
