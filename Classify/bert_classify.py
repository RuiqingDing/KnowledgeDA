#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import numpy as np
import time
import sys
sys.path.append("../")
from pytorch_pretrained import BertModel, BertTokenizer
from pytorch_pretrained.optimization import BertAdam
from sklearn import metrics
from load_data import *

parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--bert_path', type = str, default = '../PLMs/BERT_base_chinese')
parser.add_argument('--dataname', type = str, default = 'CMID')
parser.add_argument('--datatype', type = str, default = 'train')
parser.add_argument('--num_classes', type = int, default = 4)
parser.add_argument('--lr', type = float, default = 1e-5)
parser.add_argument('--epochs', type = int, default = 10)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--aug_num', type = int, default = 1)
opt = parser.parse_args()

class Config(object):
    """配置参数"""
    def __init__(self):
        self.num_classes = opt.num_classes
        self.class_list = [i for i in range(self.num_classes)]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 500                           # 若超过500 iterations 效果还没提升，则提前结束训练
        self.num_epochs = opt.epochs                                       # epoch数
        self.bert_path = opt.bert_path
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.dataname = opt.dataname
        self.datatype = opt.datatype
        self.hidden_size = 768
        self.lr = opt.lr # learning rate
        self.aug_num = opt.aug_num
        self.data_path = f'../data/{self.dataname}/{self.datatype}.txt'

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)  

    def forward(self, x):
        context = x[0]  # input sentence
        mask = x[2]
        hidden_out, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return hidden_out, out
    

start_time = time.time()

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return round(time_dif, 2)
    

# 训练model
def train(config, model, train_iter, dev_iter, learning_rate, save_file):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            hidden_outputs, outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 50 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), save_file)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, dev_iter, save_file)


# 在验证集loss达到最小后500次迭代内没有找到更小loss时，输出验证集的预测指标和混淆矩阵
def test(config, model, test_iter, save_file):
    # test
    model.load_state_dict(torch.load(save_file))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion, predict_all, probability_all = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    with open(f"result/{config.dataname}_{opt.datatype}_{config.aug_num}_{config.bert_path.split('/')[-1]}_predict_seed_{opt.seed}.txt", "w", encoding="utf-8") as f:
        f.write(str(test_report))

# 迭代后计算并打印出当前iteration上训练集的loss/accurancy和测试集的loss/accurancy
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    probability_all = []
    
    with torch.no_grad():
        for texts, labels in data_iter:
            hidden_outputs, outputs = model(texts)
            
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            
            probability = torch.softmax(outputs, dim=1)
            probability = np.round(probability.cpu().detach().numpy(), 4)
            probability_all.extend(probability)
    
    probability_all = np.array(probability_all)
    predict_all = np.argmax(probability_all, axis=1)
    acc = metrics.accuracy_score(labels_all, predict_all)

    if test:
        target_names = [str(i) for i in config.class_list]
        report = metrics.classification_report(labels_all, predict_all, target_names=target_names, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion, predict_all, probability_all
    return acc, loss_total / len(data_iter)



config = Config()
learning_rate = config.lr
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

train_iter, dev_iter, test_iter, train_num, dev_num, test_num = load_dataset(config.dataname, config.datatype, config.bert_path, config.aug_num)

# train
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
model = BERT(config).to(config.device)
    
time_train_start = time.time()
train(config, model, train_iter, dev_iter, learning_rate, f"save_dict/{config.dataname}_{config.datatype}_{config.aug_num}_{config.bert_path.split('/')[-1]}_{opt.seed}.ckpt")
print('train spending: ', get_time_dif(time_train_start))

time_test_start = time.time()
test(config, model, test_iter, f"save_dict/{config.dataname}_{config.datatype}_{config.aug_num}_{config.bert_path.split('/')[-1]}_{opt.seed}.ckpt")
print('test spending: ', get_time_dif(time_test_start))

time_dif = get_time_dif(start_time)
print("Total Time usage:", time_dif)
