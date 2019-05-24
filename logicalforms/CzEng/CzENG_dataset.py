#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:13:52 2019

@author: TiffMin
"""

import os
from sklearn.externals import joblib
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import _pickle as cPickle
from keras.preprocessing import sequence
import time


class CzENG(Dataset):
    def __init__(self, data_file, split):
        super().__init__()
        self.split = split 
        self.filepath = data_file
        self.qa_list = cPickle.load(open(self.filepath , 'rb'))
    def __len__(self):
        return len(self.qa_list)
    def __getitem__(self, idx):
        if self.split == 'test':
            return {#'glb_idx' : self.qa_list[idx]['glb_idx'],
                    
                    #'eng_tok': self.qa_list[idx]['selected_czeng_eng_moses_tok'],
                    #'cz_tok': self.qa_list[idx]['selected_czeng_cz_moses_tok'],
                    
                    'eng_leng':self.qa_list[idx]['eng_leng'],
                    'cz_leng':self.qa_list[idx]['cz_leng'],
                    
                    'eng_idxes': torch.tensor(self.qa_list[idx]['eng_idxes']),
                    'cz_idxes':torch.tensor(self.qa_list[idx]['cz_idxes'])
                    
                    
                    }
            
        else:
            random.seed(idx)
            epoch = random.sample(range(100),1)[0]
            
            random.seed(epoch*idx)
            selected_para_idx = random.sample(range(len(self.qa_list[idx]['para_idxes_lists'])), 1)[0]
            return {'glb_idx' : self.qa_list[idx]['glb_idx'],
                    
                    #'eng_tok': self.qa_list[idx]['selected_czeng_eng_moses_tok'],
                    #'cz_tok': self.qa_list[idx]['selected_czeng_cz_moses_tok'],
                    
                    'eng_leng':self.qa_list[idx]['eng_leng'],
                    'cz_leng':self.qa_list[idx]['cz_leng'],
                    
                    'eng_idxes': torch.tensor(self.qa_list[idx]['eng_idxes']),
                    'cz_idxes': torch.tensor(self.qa_list[idx]['cz_idxes']),
                    
                    #'para_tok':self.qa_list[idx]['selected_czeng_list_para_tok'][selected_para_idx],
                    'para_idxes':torch.tensor(self.qa_list[idx]['para_idxes_lists'][selected_para_idx]),
                    'para_leng': self.qa_list[idx]['para_leng_list'][selected_para_idx]
                    
                    }
            

    
class MyCollator(object):
    '''
    Yields a batch from a list of Items
    Args:
    test : Set True when using with test data loader. Defaults to False
    percentile : Trim sequences by this percentile
    '''
    def __init__(self,test=False, percentile=100, max_len = 50):
        self.test = test
        self.percentile = percentile
        #self.max_len = 50
    def __call__(self, batch):
        eng_idxes = [item['eng_idxes'] for item in batch]
        #eng_idxes = batch['eng_idxes']
        cz_idxes = [item['cz_idxes'] for item in batch]
        #cz_idxes = batch['cz_idxes']
        lens = [item['eng_leng'] for item in batch]
        max_len = np.percentile(lens, self.percentile)
        eng_idxes = sequence.pad_sequences(eng_idxes,maxlen=int(max_len), padding='post')
        #eng_idxes = torch.tensor(eng_idxes,dtype=torch.long)
        lens = [item['cz_leng'] for item in batch]
        max_len = np.percentile(lens, self.percentile)
        cz_idxes = sequence.pad_sequences(cz_idxes,maxlen=int(max_len), padding='post')
        #cz_idxes = torch.tensor(cz_idxes,dtype=torch.long)
        #batch['eng_idxes'] =eng_idxes; batch['cz_idxes'] =cz_idxes
        for k, item in enumerate(batch):
            item['eng_idxes'] = eng_idxes[k]
            item['cz_idxes'] = cz_idxes[k]
        if not self.test:
            para_idxes = [item['para_idxes'] for item in batch]
            #para_idxes = batch['para_idxes']
            lens = [item['para_leng'] for item in batch]
            max_len = max(lens)
            para_idxes = sequence.pad_sequences(para_idxes,maxlen=int(max_len), padding='post')
            #para_idxes = torch.tensor(para_idxes,dtype=torch.long)
            #batch['para_idxes'] =para_idxes
            for k, item in enumerate(batch):
                item['para_idxes'] =  para_idxes[k]

        
        return batch
