#!/usr/bin/env python
# coding: utf-8
from requests.api import get
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import numpy as np
import time
from Classify.pytorch_pretrained import BertModel, BertTokenizer
from Classify.pytorch_pretrained.optimization import BertAdam
from sklearn import metrics
from Classify.load_data import *
from Classify.aug_quality_assessment import *

parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--bert_path', type = str, default = 'BERT_base_chinese')
parser.add_argument('--dataname', type = str, default = 'CMID')
parser.add_argument('--datatype', type = str, default = 'Origin')
parser.add_argument('--num_classes', type = int, default = 4)
parser.add_argument('--lr', type = float, default = 1e-5)
parser.add_argument('--epochs', type = int, default = 10)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--aug_num', type = int, default = 1)
parser.add_argument('--delta', type=float, default = 0.75)
opt = parser.parse_args()


class Config(object):
    def __init__(self):
        self.num_classes = opt.num_classes
        self.class_list = [i for i in range(self.num_classes)]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.require_improvement = 500                          
        self.num_epochs = opt.epochs                                      
        self.bert_path = opt.bert_path
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.dataname = opt.dataname
        self.datatype = opt.datatype
        self.hidden_size = 768
        self.lr = opt.lr # learning rate
        self.aug_num = opt.aug_num
        self.delta = opt.delta

        
class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)  

    def forward(self, x):
        context = x[0]  
        mask = x[2] 
        hidden_out, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return hidden_out, out

start_time = time.time()

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return round(time_dif, 2)


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
    total_batch = 0  
    dev_best_loss = float('inf')
    last_improve = 0  
    flag = False  
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
                # dev loss no decarease in 500 iterartions???early stop
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, dev_iter, save_file)


# print classification report
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

    with open(f"Classify/result/{config.dataname}_{opt.datatype}_{config.aug_num}_{config.bert_path.split('/')[-1]}_predict_seed_{opt.seed}.txt", "w", encoding="utf-8") as f:
        f.write(str(test_report))


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
torch.backends.cudnn.deterministic = True  

df_filter = filter_text(config.dataname, config.datatype, config.aug_num, config.bert_path, config.num_classes, config.delta)
print(df_filter.head(5))

train_iter, dev_iter, test_iter, train_num, dev_num, test_num = load_filter_dataset(config.dataname, df_filter, config.bert_path, config.aug_num)

# train
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
model = BERT(config).to(config.device)
    
time_train_start = time.time()
train(config, model, train_iter, dev_iter, learning_rate, f"Classify/save_dict/{config.dataname}_{config.datatype}_Filter_{config.aug_num}_{config.bert_path.split('/')[-1]}_{opt.seed}.ckpt")
print('train spending: ', get_time_dif(time_train_start))

time_test_start = time.time()
test(config, model, test_iter, f"Classify/save_dict/{config.dataname}_{config.datatype}_Filter_{config.aug_num}_{config.bert_path.split('/')[-1]}_{opt.seed}.ckpt")
print('test spending: ', get_time_dif(time_test_start))

time_dif = get_time_dif(start_time)
print("Total Time usage:", time_dif)
