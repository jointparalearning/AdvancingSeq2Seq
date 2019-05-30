#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 18:30:45 2019

@author: TiffMin
"""

from __future__ import unicode_literals, print_function, division

from io import open
import unicodedata
import string
import re
import random
import nltk
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


######Divide lf idxes to 2nd level and 3rd level 
#These are lf indexes 
random.seed(split_num*10000)
lf_2nd_lvl_sampling = random.sample(unique_lfs_in_seventy, int(0.7*len(unique_lfs_in_seventy)))
lf_3rd_lvl_sampling = [idx for idx in unique_lfs_in_seventy if not(idx in lf_2nd_lvl_sampling)]


#####Make training_templq, validation_templq

per = 0.05
random.seed(split_num*1000)
tempqs_2nd_lvl_sampling = [lf2temp_q_dict[lf][i] for lf in lf_2nd_lvl_sampling for i in range(len(lf2temp_q_dict[lf]))]
tempqs_2nd_lvl_sampling = [tempq for tempq in tempqs_2nd_lvl_sampling if tempq in unique_templs_in_seventy]
training_templq = random.sample(tempqs_2nd_lvl_sampling, int(len(tempqs_2nd_lvl_sampling) * (1-per))) 
validation_templq = [i for i in tempqs_2nd_lvl_sampling if not(i in training_templq)]


training_templq_qidxes = [unique_templ_q_dict_seventy[templ_q][i] for templ_q in training_templq for i in range(len(unique_templ_q_dict_seventy[templ_q]))]
validation_templq_qidxes = [unique_templ_q_dict_seventy[templ_q][i] for templ_q in validation_templq for i in range(len(unique_templ_q_dict_seventy[templ_q]))]


#####Make training_lf, validation_lf
per = 0.05
random.seed(split_num*100)
qs_3rd_lvl_sampling = [unique_lf_dict_seventy[lf][i] for lf in lf_3rd_lvl_sampling for i in range(len(unique_lf_dict_seventy[lf]))]
training_lf_qidxes = random.sample(qs_3rd_lvl_sampling, int(len(qs_3rd_lvl_sampling)*(1-per)))
training_lf_qidx_dict = {q: 1 for q in training_lf_qidxes}
validation_lf_qidxes = [q for q in qs_3rd_lvl_sampling if not(q in training_lf_qidx_dict)]

training = training_templq_qidxes  + training_lf_qidxes
validation = validation_templq_qidxes + validation_lf_qidxes

assert len(training) + len(validation) == len(seventy_percent_idxes)
random.seed(split_num*10)
validation_sampled = random.sample(validation,1000)

random.seed(split_num*10)
training_sampled = random.sample(training, 10000)

print("training sampled worked")