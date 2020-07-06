#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 18:30:45 2019

@author: TiffMin
"""

##Script to sample 
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

       

#split at templ_q level
per = 0.05
random.seed(split_num*10000)
training_templq = random.sample(unique_templs_in_seventy, int(len(unique_templs_in_seventy) * (1-per))) 
validation_templq = [i for i in unique_templs_in_seventy if not(i in training_templq)]
try:
    if args.down_sample_perc < 1:
        print("downsampled by ", args.down_sample_perc)
        random.seed(split_num*10)
        training_templq_downsampled = random.sample(training_templq, int(len(training_templq)*args.down_sample_perc))
        training_templq = training_templq_downsampled
except:
    pass
            
#now at the q level 
#거의 비율 맞음
training = [unique_templ_q_dict_seventy[templ_q][i] for templ_q in training_templq for i in range(len(unique_templ_q_dict_seventy[templ_q]))]
validation = [unique_templ_q_dict_seventy[templ_q][i] for templ_q in validation_templq for i in range(len(unique_templ_q_dict_seventy[templ_q]))]
try:
    if args.down_sample_perc == 1:
        assert len(training) + len(validation) == len(seventy_percent_idxes)
except:
    pass
try:
    random.seed(split_num*1000)
    validation_sampled = random.sample(validation,1000)
    random.seed(split_num*1000)
    training_sampled = random.sample(training, 10000)
except:
    validation_sampled = validation
    training_sampled = training
