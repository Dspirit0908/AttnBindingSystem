# coding: utf-8

import json
import nltk
import torch
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from config import wikisql_path, preprocess_path


def preprocess(mode):
    """
    write [tokenize, pos_tag, lemma] to a new file. only need at beginning.
    # lemma [seems] not good for this task.
    :param mode:
    :return:
    """
    data_path, out_path = wikisql_path + mode + '.jsonl', preprocess_path + mode + '.jsonl'

    with open(data_path) as f, open(out_path, 'w') as out_f:
        for line in f:
            info = json.loads(line.strip())
            info['tokenize'] = WordPunctTokenizer().tokenize(info['question'])
            info['lemmatize'] = [WordNetLemmatizer().lemmatize(word) for word in info['tokenize']]
            info['pos_tag'] = [x[1] for x in nltk.pos_tag(info['tokenize'])]
            out_f.write(json.dumps(info) + '\n')


def get_preprocess_path(mode):
    return preprocess_path + mode + '.jsonl'


def get_wikisql_tables_path(mode):
    return wikisql_path + mode + '.tables.jsonl'


def load_data(path, only_tokenize=False):
    print('loading {}'.format(path))
    tokenize_list = []
    tokenize_len_list = []
    pos_tag_list = []
    with open(path) as f:
        for line in f:
            info = json.loads(line.strip())
            tokenize = info['tokenize']
            pos_tag = info['pos_tag']
            # check
            assert len(tokenize) == len(pos_tag)
            tokenize_list.append(tokenize)

            if only_tokenize:
                pass
            else:
                tokenize_len_list.append(len(tokenize))
                pos_tag_list.append(pos_tag)
    if only_tokenize:
        return tokenize_list
    else:
        return tokenize_list, tokenize_len_list, pos_tag_list


def build_vocab(m_lists, pre_func=None, init_vocab=None, min_count=1):
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
    new_word_count_keys = [key for key in word_count if word_count[key] >= min_count]
    sorted_wc = sorted(new_word_count_keys, key=lambda x: word_count[x], reverse=True)
    index2word = {}
    if init_vocab is None:
        word2index = {}
        num = 0
    else:
        word2index = init_vocab
        num = len(init_vocab)
        for k, v in word2index.items():
            index2word[v] = k
    word2index.update(dict(list(zip(sorted_wc, list(range(num, num + len(sorted_wc)))))))
    index2word.update(dict(list(zip(list(range(num, num + len(sorted_wc))), sorted_wc))))
    return word2index, index2word


def merge_dicts(dicts, copy=True):
    """
    :param dicts: a list of dict.
    :param copy: copy dicts[0] for merge or use dicts[0] directly.
    :return: a merged dict.
    """
    if copy:
        dict_0 = dicts[0].copy()
    else:
        dict_0 = dicts[0]
    max_value = max(dict_0.values())
    for index in range(1, len(dicts)):
        single_dict = dicts[index]
        for key, value in single_dict:
            if key in dict_0:
                pass
            else:
                max_value += 1
                dict_0[key] = max_value
    return dict_0


def set_seed(seed):
    """Set random seed everywhere."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    mode_list = ['train', 'dev', 'test']
    for mode in mode_list:
        preprocess(mode)
