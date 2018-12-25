# coding: utf-8

import sys
import time
import json
import torch
import random
import logging
import operator
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from utils import count_of_diff
from models.policy_grad import Policy
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels

np.set_printoptions(threshold=np.inf)
logger = logging.getLogger('binding')


def train(train_loader, dev_loader, args, model):
    s_time = time.time()
    print('start train... {}'.format(time.strftime('%H:%M:%S',time.localtime(time.time()))))
    if args.cuda:
        model.cuda()
    # todo: init_parameters and adjust learning rate
    # model.apply(init_parameters)
    # large_lr_layers = list(map(id, model.fc.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    best_correct = 0
    CE = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for epoch in range(1, args.epochs + 1):
        for data in train_loader:
            inputs, label, _ = data
            label = Variable(label).to(args.device)
            for i in range(len(inputs)):
                inputs[i][0] = Variable(inputs[i][0]).to(args.device)
            # zero_grad
            model.zero_grad()
            optimizer.zero_grad()
            # inference
            logit = model(inputs)
            loss = CE(logit.permute(0, 2, 1).contiguous(), label)
            # loss = 0
            # for ti in range(logit.size()[1]):
            #     loss += CE(logit[:, ti], label[:, ti])
            # loss /= logit.size()[0]
            logger.debug('loss')
            logger.debug(loss)
            loss.backward()
            optimizer.step()
            # sys.exit()
        if epoch % args.log_trian_interval == 0:
            _, _ = eval(train_loader, args, model, epoch=epoch, s_time=s_time)
        if epoch % args.log_test_interval == 0:
            correct, total = eval(dev_loader, args, model, epoch=epoch, s_time=s_time)
            if correct > best_correct and correct > 1500:
                torch.save(model, './res/' + str(correct) + '_' + time.strftime('%H-%M-%S',time.localtime(time.time())))
            best_correct = max(best_correct, correct)
    print('best correct: {}'.format(best_correct))


def eval(data_loader, args, model, epoch=None, s_time=time.time()):
    total_pred, total_true = np.array([]), np.array([])
    correct, total = 0, 0
    for data in data_loader:
        inputs, label, _ = data
        label = Variable(label).to(args.device)
        for i in range(len(inputs)):
            inputs[i][0] = Variable(inputs[i][0]).to(args.device)
        # inference
        logit = model(inputs)
        logit = torch.max(logit, 2)[1]
        pred = logit.data.cpu().numpy()
        true = label.data.cpu().numpy()
        for i in range(len(pred)):
            true_truncate, pred_truncate = [], []
            for t, p in zip(true[i], pred[i]):
                if t == -100:
                    continue
                else:
                    true_truncate.append(t), pred_truncate.append(p)
            if count_of_diff(pred_truncate, true_truncate)[0] == 0:
                correct += 1
            total += 1
            total_true, total_pred = np.append(total_true, true_truncate), np.append(total_pred, pred_truncate)
    if epoch is not None:
        print('epoch {} cost_time {}'.format(epoch, (time.time() - s_time) / 60))
    m_f1_score = f1_score(total_true, total_pred, average='micro')
    print('correct: {}, total: {}'.format(correct, total))
    print('micro f1: {}'.format(m_f1_score))
    print(classification_report(total_true, total_pred))
    print(confusion_matrix(total_true, total_pred))
    return correct, total


def train_rl(train_loader, dev_loader, args, model):
    print('start train_rl... {}'.format(time.strftime('%H:%M:%S', time.localtime(time.time()))))
    if args.cuda:
        model.cuda()
    # model.apply(init_parameters)
    # large_lr_layers = list(map(id, model.fc.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    best_correct = 0
    policy = Policy(args=args)
    # test for baseline model
    eval_rl(dev_loader, args, model, 0)
    for epoch in range(1, args.epochs + 1):
        for data in train_loader:
            # unpack data
            inputs, label, sql_label = data
            for i in range(len(inputs)):
                inputs[i][0] = Variable(inputs[i][0]).to(args.device)
            # zero_grad
            model.zero_grad()
            optimizer.zero_grad()
            # inference
            logit = model(inputs)
            tokenize_len = inputs[0][1]
            logprob, reward = policy.select_action(logit, tokenize_len, sql_label)
            loss = torch.sum(-logprob.mul(reward))
            # loss /= logit.size(0)
            loss.backward()
            optimizer.step()
        if epoch % args.log_trian_interval == 0:
            eval_rl(train_loader, args, model, epoch)
        if epoch % args.log_test_interval == 0:
            eval_rl(dev_loader, args, model, epoch)
    print('best correct: {}'.format(best_correct))


def eval_rl(data_loader, args, model, epoch):
    policy = Policy(args=args)
    rewards_epoch, step = 0, 0
    for data in data_loader:
        inputs, label, sql_label = data
        for i in range(len(inputs)):
            inputs[i][0] = Variable(inputs[i][0]).to(args.device)
        # inference
        logit = model(inputs)
        tokenize_len = inputs[0][1]
        logprob, reward = policy.select_action(logit, tokenize_len, sql_label)
        rewards_epoch += reward.mean()
        step += 1
    logger.info('reward_epoch {}'.format(str(epoch)))
    logger.info(rewards_epoch / step)
