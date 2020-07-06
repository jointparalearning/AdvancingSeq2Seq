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
    TRAIN_Shuf2Spl3OutputMasterBERTDICT = pickle.load(open('data/TRAIN_Shuf2Spl3OutputMasterBERTDICT.p', 'rb'))
    VAL_Shuf2Spl3OutputMasterBERTDICT=pickle.load( open('data/VAL_Shuf2Spl3OutputMasterBERTDICT.p', 'rb'))
    SEVENTY_Shuf2Spl3OutputMasterBERTDICT = pickle.load(open('data/SEVENTY_Shuf2Spl3OutputMasterBERTDICT.p', 'rb'))
    combinedVocab2idx = pickle.load(open('data/BERTVocab.p', "rb"))

    
    OutputMasterDictbyTypeRAW = cPickle.load(open("data/NoRepOutputMasterDictbyTypeRAW.p", "rb"))
    raw_question_binary_en = cPickle.load(open("data/raw_question_binary_ent.p", "rb"))
    
except:
    OutputMasterDictbyTypeRAW = pickle.load(open("/Volumes/Transcend/emrqa/preprocessed/OutputMaster/SMALLOutputMasterDictbyTypeRAW.p", "rb"))
    raw_question_binary_en = pickle.load(open("/Volumes/Transcend/emrqa/preprocessed/OutputMaster/SMALLraw_question_binary_ent.p", "rb"))
    
    
    
print("read files")

training_sampled = pickle.load(open('data/shuf2spl3_training_sampled.p', 'rb'))
validation_sampled = pickle.load(open('data/shuf2spl3_validation_sampled.p', 'rb'))
test_idxs = pickle.load(open('data/shuf2spl3_test_idxs.p', 'rb'))
seventy_percent_idxes = pickle.load(open('data/shuf2spl3_seventy_percent_idxes.p', 'rb'))
seventy_percent_idx_dict = {i: 'sth' for i in seventy_percent_idxes}


global_idxes_of_tokenized_eng_sentences  = seventy_percent_idxes + test_idxs

#only for idxes 
tokenized_eng_sentences = {}
for i, idx in enumerate(seventy_percent_idxes):
    tokenized_eng_sentences[idx] = SEVENTY_Shuf2Spl3OutputMasterBERTDICT['question'][i]
for i, idx in enumerate(validation_sampled):
    tokenized_eng_sentences[idx] = VAL_Shuf2Spl3OutputMasterBERTDICT['question'][i]

#only the idxes 
#[qidx for qidx, item in enumerate(tokenized_eng_sentences)]

pad_token = 0
SOS_token = 1
EOS_token = 9


Qvocab2idx = combinedVocab2idx; LFvocab2idx = combinedVocab2idx
combinedIdx2vocab = {v:k for k,v in combinedVocab2idx.items()}
Qidx2vocab = combinedIdx2vocab; LFidx2vocab = combinedIdx2vocab
combinedVocab_size = len(combinedIdx2vocab)
LFvocab_size = combinedVocab_size; Qvocab_size = combinedVocab_size
vocab_size = combinedVocab_size

def lf2idxtensor(tokenized_list, LFvocab2idx):
    vec = [LFvocab2idx[token] for token in tokenized_list]
    return vec



#Qidx2LFIdxVec_dict[0] = [0, 2, 3, 1]
#Qidx2LFIdxVec_dict_training = {qidx: lf2idxtensor(OutputMasterDictbyTypeRAW['lf'][qidx], LFvocab2idx) for qidx, item in enumerate(OutputMasterDictbyTypeRAW['lf'])}
#Qidx2TemplateIdxVec_dict = {qidx: lf2idxtensor(OutputMasterDictbyTypeRAW['templated_lf'][qidx]) for qidx, item in enumerate(OutputMasterDictbyTypeRAW['templated_lf'])}
Qidx2LFIdxVec_dict = {}
for i, idx in enumerate(seventy_percent_idxes):
    Qidx2LFIdxVec_dict[idx] = [LFvocab2idx[token] for token in SEVENTY_Shuf2Spl3OutputMasterBERTDICT['lf'][i]]
for i, idx in enumerate(validation_sampled):
    Qidx2LFIdxVec_dict[idx] = [LFvocab2idx[token] for token in VAL_Shuf2Spl3OutputMasterBERTDICT['lf'][i]]
    

def sent2idxtensor(tokenized_list, qidx):
    vec = [Qvocab2idx[token] for token in tokenized_list]
    return vec

    
print("ended loading all data!")


    