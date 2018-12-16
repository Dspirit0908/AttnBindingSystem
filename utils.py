# coding: utf-8

import json
import nltk
import torch
import random
import functools
import numpy as np
from functools import reduce
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from config import wikisql_path, preprocess_path, label_path

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


def load_data(path, only_tokenize=False, only_label=False):
    print('loading {}'.format(path))
    tokenize_list = []
    tokenize_len_list = []
    pos_tag_list = []
    table_id_list = []
    label_list = []
    with open(path) as f:
        for line in f:
            info = json.loads(line.strip())
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
        assert len(tokenize_list) == len(tokenize_len_list) == len(pos_tag_list) == len(table_id_list)
        return tokenize_list, tokenize_len_list, pos_tag_list, table_id_list, label_list


def load_tables(path, vocab=False):
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
    all_vocab = build_vocab(all_tokenize)
    return all_vocab


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
