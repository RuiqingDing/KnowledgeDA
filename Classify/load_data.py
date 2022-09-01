import torch
import time
import pandas as pd
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):
    """配置参数"""
    def __init__(self, dataname, bert_path, aug_name):
        if dataname == 'CMID':
            self.pad_size = 150     
        elif dataname == 'SO-PLC':
            self.pad_size = 200
        self.aug_name = aug_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.batch_size = 32                                      # mini-batch
        self.bert_path = bert_path
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)


PAD, CLS = '[PAD]', '[CLS]'  # padding符号

def read_data(config, df, df_type):
    '''
    读取数据
    df: 需要读取的数据
    df_type: "train", "dev", "test"
    config.pad_size: 每句话处理成的长度(短填长切)
    '''
    print("loading {}".format(df_type))
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    contents = []
    for _, row in df.iterrows():
        content = row["text"].strip()
        token = tokenizer.tokenize(content)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = tokenizer.convert_tokens_to_ids(token)

        if config.pad_size:
            if len(token) < config.pad_size:
                mask = [1] * len(token_ids) + [0] * (config.pad_size - len(token))
                token_ids += ([0] * (config.pad_size - len(token)))
            else:
                mask = [1] * config.pad_size
                token_ids = token_ids[:config.pad_size]
                seq_len = config.pad_size
    
        if df_type == "test": 
            contents.append((token_ids, seq_len, mask))
        else:
            label = eval(str(row["label"]))
            contents.append((token_ids, seq_len, mask, label))        
    return contents


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

class TestDatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # pad前的长度(超过config.pad_size的设为config.pad_size)
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, mask)
    
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_test_iterator(dataset, config):
    iter = TestDatasetIterater(dataset, config.batch_size, config.device)
    return iter


def load_dataset(dataname, datatype, bert_path, aug_num):
    config = Config(dataname, bert_path, aug_num)
    start_time = time.time()
    print("Loading data...")

    def df_read(datatype, aug_num):
        if aug_num == 1:
            print(f"data file: {dataname}/{datatype}.txt")
            df = pd.read_csv(f"../data/{dataname}/{datatype}.txt", delimiter="\t", encoding="utf-8",  header = None)
            df = df.sample(frac=1, random_state=42)  
            df.columns = ["ID", "label", "text"]

        if aug_num > 1:
            if datatype in ['test', 'dev']:
                df = pd.read_csv(f"../data/{dataname}/{datatype}.txt", delimiter="\t", encoding="utf-8",  header = None)
                df = df.sample(frac=1, random_state=42)  
                df.columns = ["ID", "label", "text"]

            elif datatype == 'knowledge':
                df1 = pd.read_csv(f"../data/{dataname}/train.txt", delimiter="\t", encoding="utf-8",  header = None) 
                df1.columns = ["ID", "label", "text"]
                df1 = df1.sample(frac=1, random_state=42)
                df2 = pd.read_csv(f"../data/{dataname}/KGER.txt", delimiter="\t", encoding="utf-8",  header = None)
                df2.columns = ["ID", "label", "text"]
                df3 = pd.read_csv(f"../data/{dataname}/TrainER.txt", delimiter="\t", encoding="utf-8",  header = None)
                df3.columns = ["ID", "label", "text"]
                tables = [df1]
                for qid in df1['ID'].tolist():
                    tables.append(df2[df2.ID==qid].sample(n=aug_num, random_state=42))
                    tables.append(df3[df3.ID==qid].sample(n=aug_num, random_state=42))
                df = pd.concat(tables, ignore_index=True)

            else:
                df1 = pd.read_csv(f"../data/{dataname}/train.txt", delimiter="\t", encoding="utf-8",  header = None) 
                df1.columns = ["ID", "label", "text"]
                df1 = df1.sample(frac=1, random_state=42)
                df2 = pd.read_csv(f"../data/{dataname}/{datatype}.txt", delimiter="\t", encoding="utf-8",  header = None)
                df2.columns = ["ID", "label", "text"]
                tables = [df1]
                for qid in df1['ID'].tolist():
                    tables.append(df2[df2.ID==qid].sample(n=aug_num, random_state=42))
                df = pd.concat(tables, ignore_index=True)

            df = df.sample(frac=1, random_state=42)  
            df.columns = ["ID", "label", "text"]
        return df
    
    print("read data")
    df_train = df_read(datatype, aug_num)
    df_dev = df_read('dev', aug_num)
    df_test = df_read('test', aug_num)
    
    print("convert")
    train_data = read_data(config, df_train , 'train')
    dev_data = read_data(config, df_dev, 'dev')
    test_data = read_data(config, df_test, 'train')
  
    print("split branch")
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    print(f'Finish loading data, spending {round(time.time() - start_time, 2)} seconds.')
    return train_iter, dev_iter, test_iter, len(df_train), len(df_dev), len(df_test)


def load_quality_assessment_data(dataname, datatype, bert_path):
    config = Config(dataname, bert_path, 1)
    start_time = time.time()
    print("Loading quality assessment data...")

    if datatype in ['KGER', 'TrainER']:
        df = pd.read_csv(f"../data/{dataname}/{datatype}.txt", delimiter="\t", encoding="utf-8",  header = None)
        df.columns = ["ID", "label", "text"]
    elif datatype == 'knowledge':
        df1 = pd.read_csv(f"../data/{dataname}/TrainER.txt", delimiter="\t", encoding="utf-8",  header = None) 
        df2 = pd.read_csv(f"../data/{dataname}/KGER.txt", delimiter="\t", encoding="utf-8",  header = None) 
        df = pd.concat([df1, df2], ignore_index=True)
        df.columns = ["ID", "label", "text"]
    else:
        print('Something wrong!!!')

    print("convert")
    train_data = read_data(config, df , 'train')
    print("split branch")
    data_iter = build_iterator(train_data, config)
    print(f'Finish loading data, spending {round(time.time() - start_time, 2)} seconds.')
    return df, data_iter


def load_filter_dataset(dataname, df, bert_path, aug_num):
    config = Config(dataname, bert_path, aug_num)
    start_time = time.time()
    print("Loading data...")

    def df_read(df_type):
        df = pd.read_csv(f"../data/{dataname}/{df_type}.txt", delimiter="\t", encoding="utf-8",  header = None)
        df = df.sample(frac=1, random_state=42)  
        df.columns = ["ID", "label", "text"]
        return df
    
    print("read data")
    df_train = df_read('train')
    df_train = pd.concat([df_train, df], ignore_index=True)
    df_train = df_train.sample(frac=1, random_state=42) 
    df_dev = df_read('dev')
    df_test = df_read('test')
    
    print("convert")
    train_data = read_data(config, df_train , 'train')
    dev_data = read_data(config, df_dev, 'dev')
    test_data = read_data(config, df_test, 'train')
  
    print("split branch")
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    print(f'Finish loading data, spending {round(time.time() - start_time, 2)} seconds.')

    return train_iter, dev_iter, test_iter, len(df_train), len(df_dev), len(df_test)

# if __name__ == "__main__":
#     train_iter, dev_iter, test_iter, train_num, dev_num, test_num = load_dataset('Abstract', 'Origin','PLMs/BERT_base')