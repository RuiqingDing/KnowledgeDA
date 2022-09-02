# KnowledgeDA

## KnowledgeDA Framework

![KnowledgeDA](img/KnowledgeDA_framework_v3.png)

## Requirements

* Python 3.6
* torch 1.8
* cuda 1.11
* nltk 3.6
* jieba  0.42.1
* synonyms 3.16.0
* scikit-learn 0.24.1
* ordered_set 4.0.2


## Quick Start

### Data & PLMs Preparation

* Data: unzip the files in `data` and  `KG`

* PLMs: download PLMs from https://huggingface.co/ and put them into `PLMs`

  * BERT_base_chinese: https://huggingface.co/bert-base-chinese
  
  * BERT_base: https://huggingface.co/bert-base-uncased


### Run Service

* **Example 1: CMID (Chinese)**

```
python run_service.py --dataname CMID --domain healthcare --lan ch --plm_path BERT_base_chinese
```

* **Example 2: SO-PLC (English)**

```
python run_service.py --dataname SO-PLC --domain software --lan en --plm_path BERT_base
```

Attention: the fine-tuned model is saved in `Classify/save_dict.`


### Extension to other task data

* Put your task data into `data` folder

  * include 3 txt files: train, dev and test datasets
  
  * each row in train.txt and dev.txt should be like: `[id]\t[label]\t[text]`
  
  * can refer to the dataset CMID and SO-PLC
  
* put your domain KG into `KG` folder

  * include 2 main files: triples and the dictionary of entity categories
  
  * can refer to the given KG format
  

