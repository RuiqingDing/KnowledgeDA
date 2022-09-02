from Augment.augment_kg_ch import *
from Augment.augment_kg_en import *
from Classify.bert_classify import *
import argparse

parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--plm_path', type = str, default = 'BERT_base_chinese')
parser.add_argument('--dataname', type = str, default = 'CMID')
parser.add_argument('--datatype', type = str, default = 'knowledge')
parser.add_argument('--num_classes', type = int, default = 4)
parser.add_argument('--lr', type = float, default = 1e-5)
parser.add_argument('--epochs', type = int, default = 10)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--aug_num', type = int, default = 5)
parser.add_argument('--domain', type = str, default = 'healthcare') # domain
parser.add_argument('--lan', type = str, default = 'ch') # language
opt = parser.parse_args()

# Step1: augment data
if opt.lan == 'ch':
    if opt.domain == 'healthcare':
        augment_KnowledgeDA_ch(opt.dataname) # using CMedicalKG
        # Step2: train classification model
        train_knowledgeDA(opt.plm_path, opt.dataname, opt.datatype, opt.aug_num, opt.num_classes, opt.epochs, opt.seed)

    else:
        print(f'For {opt.domain}({opt.lan}): No suitable build-in KG, please add KG data into KG folder')
        print('Stop the service ...')

elif opt.lan == 'en':
    if opt.domain == 'software':
        augment_KnowledgeDA_en(opt.dataname)
        # Step2: train classification model
        train_knowledgeDA(opt.plm_path, opt.dataname, opt.datatype, opt.aug_num, opt.num_classes, opt.epochs, opt.seed)

    else:
        print(f'For {opt.doamin}({opt.lan}): No suitable build-in KG, please add KG data into KG folder')
        print('Stop the service ...')



