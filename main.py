# coding: utf-8

import torch
import functools
from utils import get_preprocess_path, get_wikisql_tables_path, load_data, build_vocab


def main():
    # need to know all the words to filter the pretrained word embeddings
    load_only_tokenize = functools.partial(load_data, only_tokenize=True)
    mode_list = ['train', 'dev', 'test']
    all_tokenize = []
    for mode in mode_list:
        tokenize = load_only_tokenize(get_preprocess_path(mode))
        all_tokenize.extend(tokenize)
    all_vocab = build_vocab(all_tokenize)


if __name__ == '__main__':
    main()
