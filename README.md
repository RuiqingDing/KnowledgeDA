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

* **After unzipping, the file directory is as follows**

 ```
 .
 ├── Augment
 │   ├── augment_kg_ch.py
 │   ├── augment_kg_en.py
 │   └── utils
 │       ├── cluster_ch.py
 │       ├── cluster_en.py
 │       ├── entity_mask_ch.py
 │       ├── entity_mask_en.py
 │       ├── file_handler.py
 │       ├── load_data_ch.py
 │       └── load_data_en.py
 ├── Classify
 │   ├── aug_quality_assessment.py
 │   ├── bert_classification.py
 │   ├── BERT_KnowledgeDA.py
 │   ├── load_data.py
 │   ├── pytorch_pretrained
 │   ├── result
 │   └── save_dict
 │
 ├── data
 │   ├── CMID
 │   │   ├── dev.txt
 │   │   ├── test.txt
 │   │   └── train.txt
 │   └── SO-PLC
 │       ├── dev.txt
 │       ├── test.txt
 │       └── train.txt
 ├── KG
 │   ├── CMedicalKG
 │   │   ├── entities_dict.txt
 │   │   ├── entities_list.txt
 │   │   └── triples.txt
 │   └── TagKG
 │       ├── entities_dict.txt
 │       ├── entities_list.txt
 │       └── triples.csv
 ├── PLMs
 │   ├── BERT_base
 │   └── BERT_base_chinese
 ├── run_service.py

 ```


### Run Service

* **Example 1: CMID (Chinese)**

```
python run_service.py --dataname CMID --domain healthcare --lan ch --plm_path BERT_base_chinese
```

* **Example 2: SO-PLC (English)**

```
python run_service.py --dataname SO-PLC --domain software --lan en --plm_path BERT_base
```

* **Attention**

  * the fine-tuned model will be saved in `Classify/save_dict`
  * the result will be store as the classification report in `Classify/result`, here we give an example of the result on CMID

    ```
                    precision    recall  f1-score   support

                 0     0.8061    0.8421    0.8237       627
                 1     0.7544    0.7681    0.7612       276
                 2     0.5176    0.5665    0.5410       233
                 3     0.6571    0.2556    0.3680        90

          accuracy                         0.7300      1226
         macro avg     0.6838    0.6081    0.6235      1226
      weighted avg     0.7287    0.7300    0.7225      1226
    ```

### Extension to other task data

If your task is related to healthcare (Chinese) or software (English), you can directly use our built-in KG; otherwise, you can also use your own KG:

* Put your task data into `data`
  * include 3 txt files: train, dev and test datasets
  * each row in train.txt and dev.txt should be like: `[id]\t[label]\t[text]`
  * can refer to the dataset CMID and SO-PLC
  
* put your domain KG into `KG`
  * include 2 main files: triples and the dictionary of entity category
  * can refer to the given KG format
  

