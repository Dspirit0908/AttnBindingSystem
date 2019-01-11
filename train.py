# coding: utf-8

import sys
import time
import json
import torch
import random
import logging
import datetime
import operator
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.policy_grad import Policy
from utils import count_of_diff, translate_m_lists, sequence_mask
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

np.set_printoptions(threshold=np.inf)
logger = logging.getLogger('binding')


def train(train_loader, dev_loader, args, model):
    s_time = time.time()
    print('start train... {}'.format(time.strftime('%H:%M:%S',time.localtime(time.time()))))
    if args.cuda:
        model.cuda()
    # todo: init_parameters and adjust learning rate
    # model.apply(init_parameters)
    freeze_lr_layers = list(map(id, model.token_embedding.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    best_correct = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for epoch in range(1, args.epochs + 1):
        for data in train_loader:
            inputs, (label, _), _ = data
            label = Variable(label).to(args.device)
            for i in range(len(inputs)):
                inputs[i][0] = Variable(inputs[i][0]).to(args.device)
            # zero_grad
            model.zero_grad()
            optimizer.zero_grad()
            if args.crf:
                loss = model.forward_loss(inputs, label)
            else:
                # feed forward
                _, _, logit = model(inputs)
                # loss = criterion(logit.permute(0, 2, 1).contiguous(), label)
                loss = 0
                for ti in range(logit.size()[1]):
                    loss += criterion(logit[:, ti], label[:, ti])
            loss.backward()
            optimizer.step()
            # sys.exit()
        if epoch % args.log_trian_interval == 0:
            _, _ = eval(train_loader, args, model, epoch=epoch, s_time=s_time)
        if epoch % args.log_test_interval == 0:
            correct, total = eval(dev_loader, args, model, epoch=epoch, s_time=s_time)
            if correct > best_correct and correct / total > args.save_bar_pretrained:
                model_path = './res/' + args.model + '/' + str(correct) + '_' + str(args.cell_info) + '_' + str(args.attn_concat) + '_' + str(args.crf) + '_' + str(datetime.datetime.now().microsecond)
                torch.save(model, model_path)
                logger.info('save model: {}'.format(model_path))
            best_correct = max(best_correct, correct)
        logger.info('- epoch {}, best correct: {}'.format(str(epoch), best_correct))


def eval(data_loader, args, model, epoch=None, s_time=time.time()):
    total_pred, total_true = np.array([]), np.array([])
    correct, total = 0, 0
    for data in data_loader:
        inputs, (label, _), _ = data
        label = Variable(label).to(args.device)
        for i in range(len(inputs)):
            inputs[i][0] = Variable(inputs[i][0]).to(args.device)
        # feed forward
        _, _, logit = model(inputs)
        if args.crf:
            raise NotImplementedError
        else:
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
    policy = Policy(args=args)
    # test for baseline model
    best_correct_ratio = eval_rl(dev_loader, args, model, epoch=0)
    for epoch in range(1, args.epochs + 1):
        for data in train_loader:
            # unpack data
            inputs, (label, _), sql_labels = data
            for i in range(len(inputs)):
                inputs[i][0] = Variable(inputs[i][0]).to(args.device)
            # zero_grad
            model.zero_grad()
            optimizer.zero_grad()
            # feed forward
            _, _, logit = model(inputs)
            tokenize_len = inputs[0][1]
            m_log_probs, m_rewards = policy.select_m_actions(logit, tokenize_len, sql_labels)
            loss = torch.sum(-m_log_probs.mul(m_rewards))
            # loss /= logit.size(0)
            logger.info('reward_mean')
            logger.info(m_rewards.mean())
            logger.info('loss_mean')
            logger.info(loss / logit.size(0))
            loss.backward()
            optimizer.step()
        if epoch % args.log_trian_interval == 0:
            _ = eval_rl(train_loader, args, model, epoch)
        if epoch % args.log_test_interval == 0:
            correct_ratio = eval_rl(dev_loader, args, model, epoch)
            if correct_ratio > best_correct_ratio and correct_ratio > args.save_bar_rl:
                model_path = './res/policy_gradient/' + str(correct_ratio.data.cpu().numpy()) + '_' + str(args.cell_info) + '_' + str(args.attn_concat) + '_' + str(args.crf) + '_' + str(datetime.datetime.now().microsecond)
                torch.save(model, model_path)
                logger.info('save model: {}'.format(model_path))
            best_correct_ratio = max(best_correct_ratio, correct_ratio)
        logger.info('- epoch {}, best correct ratio: {}'.format(str(epoch), best_correct_ratio))


def eval_rl(data_loader, args, model, epoch):
    policy = Policy(args=args)
    rewards_epoch, total_batch = 0, 0
    t_error_1, t_error_2, t_error_3, t_error_4 = 0, 0, 0, 0
    for data in data_loader:
        inputs, (label, _), sql_labels = data
        for i in range(len(inputs)):
            inputs[i][0] = Variable(inputs[i][0]).to(args.device)
        # feed forward
        _, _, logit = model(inputs)
        tokenize_len = inputs[0][1]
        actions, reward, b_error_1, b_error_2, b_error_3, b_error_4 = policy.select_max_action(logit, tokenize_len, sql_labels)
        rewards_epoch += reward.sum()
        total_batch += reward.size(0)
        t_error_1 += b_error_1
        t_error_2 += b_error_2
        t_error_3 += b_error_3
        t_error_4 += b_error_4
    logger.info('reward_epoch {}'.format(str(epoch)))
    logger.info(rewards_epoch / total_batch)
    print('error')
    print(t_error_1, t_error_2, t_error_3, t_error_4)
    print('error_ratio')
    print(t_error_1 / total_batch, t_error_2 / total_batch, t_error_3 / total_batch, t_error_4 / total_batch)
    return rewards_epoch / total_batch


def test(data_loader, args, model, sep=''):
    res = {}
    policy = Policy(args=args)
    for data in data_loader:
        inputs, (label, _), sql_labels = data
        for i in range(len(inputs)):
            inputs[i][0] = Variable(inputs[i][0]).to(args.device)
        # feed forward
        _, _, logit = model(inputs)
        tokenize_len = inputs[0][1]
        actions, _, _, _, _, _ = policy.select_max_action(logit, tokenize_len, sql_labels)
        questions = translate_m_lists(inputs[0][0].data.cpu().numpy(), the_dict=args.index2word, sep=sep)
        batch_sel_col, batch_conds_cols, batch_conds_values = sql_labels
        batch_sel_col, batch_conds_cols, batch_conds_values = batch_sel_col.data.cpu().numpy().tolist(), batch_conds_cols.data.cpu().numpy().tolist(), batch_conds_values.data.cpu().numpy().tolist()
        for index, (question, action) in enumerate(zip(questions, actions.data.cpu().numpy().tolist())):
            key = question.replace('<unk>', '')
            value = action[:tokenize_len[index]]
            if key not in res:
                res[key] = {}
            res[key]['pred'] = value
            res[key]['label'] = label.data.cpu().numpy().tolist()[index][:tokenize_len[index]]
            res[key]['sql_labels'] = [batch_sel_col[index], list(batch_conds_cols[index]), list(batch_conds_values[index])]
    return res
