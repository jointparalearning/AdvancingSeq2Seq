#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:31:58 2018

@author: TiffMin
"""

from __future__ import unicode_literals, print_function, division

from io import open
import unicodedata
import string
import re
import random
#import nltk
import numpy as np
import pickle#, dill
import _pickle as cPickle
import math,sys, copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.autograd import Variable
import numpy as np
import copy

try:
    OutputMasterDictbyTypeRAW = cPickle.load(open("data/NoRepOutputMasterDictbyTypeRAW.p", "rb"))
    vocab2idx_dictRAW = cPickle.load(open("data/NoRepUnique_idx2vocab_dictRAW.p", "rb"))
    raw_question_binary_en = cPickle.load(open("data/raw_question_binary_ent.p", "rb"))
    
except:
    
    OutputMasterDictbyTypeRAW = pickle.load(open("/Volumes/Transcend/emrqa/preprocessed/OutputMaster/SMALLOutputMasterDictbyTypeRAW.p", "rb"))
    vocab2idx_dictRAW = pickle.load(open("/Volumes/Transcend/emrqa/preprocessed/OutputMaster/unique_idx2vocab_dictRAW.p", "rb"))
    raw_question_binary_en = pickle.load(open("/Volumes/Transcend/emrqa/preprocessed/OutputMaster/SMALLraw_question_binary_ent.p", "rb"))
    
print("read files")

tokenized_eng_sentences = OutputMasterDictbyTypeRAW['question']
global_idxes_of_tokenized_eng_sentences  = [qidx for qidx, item in enumerate(OutputMasterDictbyTypeRAW['question'])]

pad_token = 0
SOS_token = 1
EOS_token = 2
Qvocab2idx = vocab2idx_dictRAW['question']
LFvocab2idx = vocab2idx_dictRAW['lf']

combinedVocab2idx = copy.deepcopy(Qvocab2idx)
new_count = len(combinedVocab2idx)
for k, v in LFvocab2idx.items():
    if not(k in Qvocab2idx):
        combinedVocab2idx[k] = new_count
        new_count+=1

Qvocab2idx = combinedVocab2idx; LFvocab2idx = combinedVocab2idx
combinedIdx2vocab = {v:k for k,v in combinedVocab2idx.items()}
Qidx2vocab = combinedIdx2vocab; LFidx2vocab = combinedIdx2vocab
combinedVocab_size = len(combinedIdx2vocab)
LFvocab_size = combinedVocab_size; Qvocab_size = combinedVocab_size
vocab_size = combinedVocab_size

def lf2idxtensor(tokenized_list, LFvocab2idx):
    vec = [SOS_token]+[LFvocab2idx[token] for token in tokenized_list]+[EOS_token]
    return vec


#Qidx2LFIdxVec_dict[0] = [0, 2, 3, 1]
Qidx2LFIdxVec_dict = {qidx: lf2idxtensor(OutputMasterDictbyTypeRAW['lf'][qidx], LFvocab2idx) for qidx, item in enumerate(OutputMasterDictbyTypeRAW['lf'])}
#Qidx2TemplateIdxVec_dict = {qidx: lf2idxtensor(OutputMasterDictbyTypeRAW['templated_lf'][qidx]) for qidx, item in enumerate(OutputMasterDictbyTypeRAW['templated_lf'])}

print("loaded all preprocessing")


##Device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



def sent2idxtensor(tokenized_list, qidx):
    vec = [SOS_token]+[Qvocab2idx[token] for token in tokenized_list]+[EOS_token]
    return vec


#Pick test set first
random.seed(split_num*100000)
try:
    seventy_percent_idxes = random.sample(global_idxes_of_tokenized_eng_sentences, int(len(tokenized_eng_sentences)*0.7+138000)) #[np.random.choice(global_idxes_of_tokenized_eng_sentences) for i in range(int(len(tokenized_eng_sentences)*0.7))]
except:
    seventy_percent_idxes = random.sample(global_idxes_of_tokenized_eng_sentences, int(len(tokenized_eng_sentences)*0.7)) #[np.random.choice(global_idxes_of_tokenized_eng_sentences) for i in range(int(len(tokenized_eng_sentences)*0.7))]
seventy_percent_idx_dict = {i: 'sth' for i in seventy_percent_idxes}
test_idxs = [i for i in global_idxes_of_tokenized_eng_sentences if i not in seventy_percent_idx_dict]
prop = 32*5*2000/len(seventy_percent_idx_dict)
random.seed((split_num*1000000))
try:
    test_idxs = random.sample(test_idxs,int(len(test_idxs)*prop))
except:
    pass


#####################USEFUL DICTIONARIES##########################
#Find all unique templated question indexes
templ_qlist = OutputMasterDictbyTypeRAW['templated_question']
templ_lflist = OutputMasterDictbyTypeRAW['templated_lf']

#how the unique dictionary works:
#{7: 1,3,5}이면, so_far_list[7]과 templ_qlist[1], templ_qlist[3], templ_qlist[5]가 같은 templated question이라는 것임
unique_templ_q_dict = {}
so_far_list = []
so_far_idx = 0
for idx in range(len(templ_qlist)):
    #print(idx)
    if not(templ_qlist[idx] in so_far_list):
        so_far_list.append(templ_qlist[idx])
        unique_templ_q_dict[so_far_idx] = [idx]
        so_far_idx +=1
    else:
        far_idx = so_far_list.index(templ_qlist[idx])
        #print("far idx" + str(far_idx))
        unique_templ_q_dict[far_idx].append(idx)

rev_unique_templ_q_dict = {vv: k for k, v in unique_templ_q_dict.items() for vv in v}


##Only for seventy idxes
unique_templs_in_seventy = list(set([rev_unique_templ_q_dict[qidx] for qidx in seventy_percent_idxes]))
rev_unique_templ_q_dict_seventy = {k: rev_unique_templ_q_dict[k] for k in seventy_percent_idxes} 
unique_templ_q_dict_seventy = {}
for k, q_list in unique_templ_q_dict.items():
    if k in unique_templs_in_seventy:
        unique_templ_q_dict_seventy[k] = [q for q in q_list if q in seventy_percent_idx_dict]


###Make training_lf, validation_lf
unique_lf_dict = {}
lf_so_far_list = []
lf_so_far_idx = 0
for idx in range(len(templ_lflist)):
    #print(idx)
    if not(templ_lflist[idx] in lf_so_far_list):
        lf_so_far_list.append(templ_lflist[idx])
        unique_lf_dict[lf_so_far_idx] = [idx]
        lf_so_far_idx +=1
        #lf_2_templq_dict[lf_so_far_idx] = [templ_qlist[idx]]
    else:
        far_idx = lf_so_far_list.index(templ_lflist[idx])
        #print("far idx" + str(far_idx))
        unique_lf_dict[far_idx].append(idx)
        #lf_2_templq_dict[far_idx].append(idx)

rev_unique_lf_dict = {vv: k for k, v in unique_lf_dict.items() for vv in v}

##Only for seventy idxes
unique_lfs_in_seventy = list(set([rev_unique_lf_dict[qidx] for qidx in seventy_percent_idxes]))
rev_unique_lf_dict_seventy = {k: rev_unique_lf_dict[k] for k in seventy_percent_idxes} 
unique_lf_dict_seventy = {}
for k, q_list in unique_lf_dict.items():
    if k in unique_lfs_in_seventy:
        unique_lf_dict_seventy[k] = [q for q in q_list if q in seventy_percent_idx_dict]

##lf 2 templq dictionary 
lf2temp_q_dict = {}
count = 0
for i in range(lf_so_far_idx): 
    current_qs = unique_lf_dict[i]
    current_qs_2_templ_qs = [rev_unique_templ_q_dict[c] for c in current_qs]
    current_qs_2_templ_qs  = list(set(current_qs_2_templ_qs))
    lf2temp_q_dict[i] = current_qs_2_templ_qs
    count += len(current_qs_2_templ_qs)


rev_lf2temp_q_dict = {vv: k for k, v in lf2temp_q_dict.items() for vv in v}

print("loaded all dictionaries")

#################################################################


#Pick the validation here 
if shuffle_scheme == 0:
    if len(seventy_percent_idxes) > 1000:
        val_len = int(0.05*len(seventy_percent_idxes))
        validation = seventy_percent_idxes[:val_len]
        validation_sampled = seventy_percent_idxes[:1000]
        training = seventy_percent_idxes[val_len:]
        training_sampled = training[:10000]
    else:
        validation_sampled = seventy_percent_idxes[:100]
        training = seventy_percent_idxes[100:]
if shuffle_scheme == 1:
    exec(open('data_prep/shuffle.py').read())
elif shuffle_scheme == 2:
    exec(open('data_prep/2nd_shuffle.py').read())
    
print("ended loading all data!")


    