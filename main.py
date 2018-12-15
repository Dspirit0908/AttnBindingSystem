# coding: utf-8

import torch
import functools
from utils import get_preprocess_path, get_wikisql_tables_path, load_data, load_tables, build_vocab


def main():
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


if __name__ == '__main__':
    main()
