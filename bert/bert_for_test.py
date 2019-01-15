# coding: utf-8

from pytorch_pretrained_bert import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print(tokenizer.tokenize('+14.0'))
