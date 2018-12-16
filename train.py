# coding: utf-8

import time
import json
import torch
import random
import operator
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torchcrf import CRF
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
np.set_printoptions(threshold=np.inf)


def train(train_loader, dev_loader, args, model):
    s_time = time.time()
    if args.cuda:
        model.cuda()
    # large_lr_layers = list(map(id, model.fc.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    best_correct = 0
    for epoch in range(1, args.epochs + 1):
        for data in train_loader:
            inputs, label = data
            label = Variable(label)
            if args.cuda:
                label = label.cuda()
            for i in range(len(inputs)):
                inputs[i][0] = Variable(inputs[i][0])
                if args.cuda:
                    inputs[i][0] = inputs[i][0].cuda()

            model.zero_grad()
            logit = model(inputs)
            print(logit.size())
            print(label.size())
            loss = F.cross_entropy(logit, label)
            loss.backward()
            optimizer.step()
        
        if epoch % args.log_trian_interval == 0:
            _, _ = eval(train_loader, args, model, epoch=epoch, s_time=s_time)
        if epoch % args.log_test_interval == 0:
            correct, total = eval(dev_loader, args, model, epoch=epoch, s_time=s_time)
            best_correct = max(best_correct, correct)
    print('best correct: {}'.format(best_correct))


def eval(data_loader, args, model, epoch=None, s_time=time.time()):
    total_pred, total_true = np.array([]), np.array([])
    correct, total = 0, 0
    for data in data_loader:
        inputs, label = data
        label = Variable(label)
        if args.cuda:
            label = label.cuda()
        for i in range(len(inputs)):
            inputs[i][0] = Variable(inputs[i][0])
            if args.cuda:
                inputs[i][0] = inputs[i][0].cuda()

        logit = model(inputs)
        pred = torch.max(logit, 1)[1].data.cpu().numpy()
        true = label.data.cpu().numpy()
        total_pred, total_true = np.append(total_pred, pred), np.append(total_true, true)
    if epoch is not None:
        print('epoch {} cost_time {}'.format(epoch, (time.time() - s_time) / 60))
    m_f1_score = f1_score(total_true, total_pred, average='macro')
    print('correct: {}, total: {}'.format(correct, total))
    print('macro f1: {}'.format(m_f1_score))
    print(classification_report(total_true, total_pred))
    print(confusion_matrix(total_true, total_pred))
    return correct, total
