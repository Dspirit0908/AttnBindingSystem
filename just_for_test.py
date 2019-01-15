# coding: utf-8

import os
import json
import time
import torch
import random
import datetime
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "Who was Jim Henson ? [SEP] Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print(indexed_tokens)
# bert = BertModel.from_pretrained('bert-base-uncased')
# print(bert.config.hidden_size)
