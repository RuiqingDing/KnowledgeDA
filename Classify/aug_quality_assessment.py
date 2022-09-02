import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from Classify.pytorch_pretrained import BertModel, BertTokenizer
from Classify.pytorch_pretrained.optimization import BertAdam
from sklearn import metrics
from Classify.load_data import *


class Config(object):
    def __init__(self, dataname, datatype, aug_num, bert_path, num_classes, delta):
        self.num_classes = num_classes
        self.class_list = [i for i in range(self.num_classes)]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.bert_path = bert_path
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.dataname = dataname
        self.datatype = datatype
        self.hidden_size = 768
        self.aug_num = aug_num
        self.delta = delta

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
        pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out

start_time = time.time()

def softmax(x):
    row_max = np.max(x) 
    x = x - row_max 
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp)
    s = x_exp / x_sum
    return s

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return round(time_dif, 2)

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
    return probability_all 


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    probability_all = []
    
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
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


def filter_text(dataname, datatype, aug_num, bert_path, num_classes, delta):
    config = Config(dataname, datatype, aug_num, bert_path, num_classes, delta)

    df, data_iter = load_quality_assessment_data(config.dataname, config.datatype, config.bert_path)    
    model = BERT(config).to(config.device)

    time_test_start = time.time()
    probability_all  = test(config, model, data_iter, f"Classify/save_dict/{config.dataname}_train_1_{config.bert_path.split('/')[-1]}_1.ckpt")
    print('test spending: ', get_time_dif(time_test_start))

    for index, row in df.iterrows(): #df.columns = ["ID", "label", "text"]
        label = int(row['label'])
        df.loc[index, 'prob'] = probability_all[index][label]

    df['prob'] = 1 - np.abs(config.delta - df['prob'])
    tables = []
    for sid in df['ID'].unique():
        df_sub = df[df.ID == sid].reset_index()
        x = list(softmax(df_sub['prob'].tolist()))
        df_sub['weight'] = x
        df_sub_sample = df_sub.sample(config.aug_num, weights='weight',replace=False)
        tables.append(df_sub_sample)

    df_sample = pd.concat(tables, ignore_index=True)
    df_sample = df_sample[["ID", "label", "text"]]

    return df_sample

