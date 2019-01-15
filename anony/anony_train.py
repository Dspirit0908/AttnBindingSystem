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
from torch.nn.utils import clip_grad_norm
from models.policy_grad import Policy
from tensorboardX import SummaryWriter
from utils import count_of_diff, translate_m_lists, sequence_mask, BOS_WORD, EOS_WORD
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

np.set_printoptions(threshold=np.inf)
logger = logging.getLogger('binding')


def train(train_loader, dev_loader, args, model):
    s_time = time.time()
    print('start train... {}'.format(time.strftime('%H:%M:%S',time.localtime(time.time()))))
    log_dir = './logs/' + str(args.cell_info) + '_' + str(args.attn_concat) + '_' + str(args.crf) + '_' + str(datetime.datetime.now().microsecond)
    logger.info(log_dir)
    dev_writer = SummaryWriter(log_dir=log_dir)
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    best_correct = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0
        for data in train_loader:
            inputs, label = data
            label = Variable(label).to(args.device)
            for i in range(len(inputs)):
                inputs[i][0] = Variable(inputs[i][0]).to(args.device)
            # zero_grad
            model.zero_grad()
            optimizer.zero_grad()
            inputs = inputs.transpose(0, 1).contiguous()
            label = label.transpose(0, 1).contiguous()
            output = model(inputs, label)
            loss = F.nll_loss(output[1:].contiguous().view(-1, len(args.anony_query_vocab)),
                              label[1:].contiguous().view(-1),
                              ignore_index=args.anony_query_vocab[EOS_WORD])
            epoch_loss += loss
            clip_grad_norm(model.parameters(), 10)
            loss.backward()
            optimizer.step()
            # sys.exit()
        logger.info('epoch {} epoch_loss: {}'.format(str(epoch), epoch_loss))
        # if epoch % args.log_trian_interval == 0:
        #     _, _ = eval(train_loader, args, model, epoch=epoch, s_time=s_time)
        #     model_path = './res/' + args.model + '/' + str(args.cell_info) + '_' + str(args.attn_concat) + '_' + str(args.crf) + '_' + str(datetime.datetime.now().microsecond) + '_' + str(epoch)
        #     torch.save(model, model_path)
        #     logger.info('save model: {}'.format(model_path))
        if epoch % args.log_test_interval == 0:
            correct, total = eval(dev_loader, args, model, epoch=epoch, s_time=s_time)
        #     dev_writer.add_scalar('Correct_Ratio', correct / total, epoch)
        #     if correct > best_correct and correct / total > args.save_bar_pretrained:
        #         model_path = './res/' + args.model + '/' + str(correct) + '_' + str(args.cell_info) + '_' + str(args.attn_concat) + '_' + str(args.crf) + '_' + str(datetime.datetime.now().microsecond)
        #         torch.save(model, model_path)
        #         logger.info('save model: {}'.format(model_path))
            best_correct = max(best_correct, correct)
        logger.info('- epoch {}, best correct: {}'.format(str(epoch), best_correct))


def eval(data_loader, args, model, epoch=None, s_time=time.time()):
    total_pred, total_true = np.array([]), np.array([])
    correct, total = 0, 0
    for data in data_loader:
        inputs, label = data
        label = Variable(label).to(args.device)
        for i in range(len(inputs)):
            inputs[i][0] = Variable(inputs[i][0]).to(args.device)
        # feed forward
        inputs = inputs.transpose(0, 1).contiguous()
        label = label.transpose(0, 1).contiguous()
        output = model(inputs, label, teacher_forcing_ratio=0.0)
        output = output.transpose(0, 1).contiguous()
        label = label.transpose(0, 1).contiguous()
        logit = torch.max(output, -1)[1]
        pred = logit.data.cpu().numpy().tolist()
        true = label.data.cpu().numpy().tolist()
        for i in range(len(pred)):
            true_truncate, pred_truncate = [], []
            for t, p in zip(true[i], pred[i]):
                if t == args.anony_query_vocab[EOS_WORD] or t == args.anony_query_vocab[BOS_WORD]:
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
