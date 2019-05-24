#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:05:26 2019

@author: TiffMin
"""

domain = 'blocks'
#domain = 'calendar'
#domain = 'basketball'
#domain = 'recipes'
#domain = 'housing'
#domain = 'publications'
#domain = 'socialnetwork'
#domain = 'restaurants'

train_file_name = '/Volumes/Transcend/fewshot learning/overnight-lf/' + domain + '_train.tsv'
test_file_name = '/Volumes/Transcend/fewshot learning/overnight-lf/' + domain + '_test.tsv'

line_list = []
with open(train_file_name) as open_f:
    for i, line in enumerate(open_f):
        line_list.append(line)

train_list = {}
for k, line in enumerate(line_list):
    line = line.strip('\n')
    line  = line.split('\t')
    train_list[k] = {}
    train_list[k]['utterance'], train_list[k]['answer'] = line[0], line[1].replace('SW.', '').replace('en.', '') 


line_list = []
with open(test_file_name) as open_f:
    for i, line in enumerate(open_f):
        line_list.append(line)

test_list = {}
for k, line in enumerate(line_list):
    line = line.strip('\n')
    line  = line.split('\t')
    test_list[k] = {}
    test_list[k]['utterance'], test_list[k]['answer'] = line[0], line[1].replace('SW.', '').replace('en.', '') 

###Map to paraphrase groups

para_file_name = '/Volumes/Transcend/overnightData/' + domain + '.paraphrases.groups.txt'

line_list = []
with open(para_file_name) as open_f:
    for i, line in enumerate(open_f):
        line_list.append(line)
        
original_idx_dict = {}
para_group_dict = {}
para_utterance_idx_dict = {}
for i, line in enumerate(line_list):
    temp_list = line.split('-')
    ori_para = temp_list[0].strip()
    utt = temp_list[1].strip()
    if ori_para == 'original':
        original_idx_dict[len(original_idx_dict)] = utt
        para_group_dict[len(para_group_dict)] = []
    else:
        utt = utt.replace(', worker', '')
        para_utterance_idx_dict[len(para_utterance_idx_dict)] = utt
        para_group_num = len(para_group_dict) -1 ; para_utt_num = len(para_utterance_idx_dict)-1
        para_group_dict[para_group_num].append(para_utt_num)

global_idx_dict = {}
global_idx_dict['utterance'] = para_utterance_idx_dict
global_idx_dict['para_group'] = {vv:k for k, v in para_group_dict.items() for vv in v}

#Match global idxes to existing match 
global_idx_2_gold_train_idx_dict = {}
gold_train_2_global_idx_dict = {}
global_idx_2_gold_test_idx_dict = {}
gold_test_2_global_idx_dict = {}
para_lf_idx_dict = {}

weird_para_proc_utt = []
proc_utt_dict = {}
for i, utt in para_utterance_idx_dict.items():
    proc_utt = ''.join(utt.split(' '))
    if proc_utt in proc_utt_dict:
        weird_para_proc_utt.append(i)
    proc_utt_dict[proc_utt] = i

###############
#Back to train and test
for k, t_d in train_list.items():
    glb_idx = proc_utt_dict[''.join(train_list[k]['utterance'].split(' '))]
    t_d['glb_idx'] = glb_idx 
    t_d['para_group'] = global_idx_dict['para_group'][glb_idx]
    
for k, t_d in test_list.items():
    glb_idx = proc_utt_dict[''.join(test_list[k]['utterance'].split(' '))]
    t_d['glb_idx'] = glb_idx 
    t_d['para_group'] = global_idx_dict['para_group'][glb_idx]
 
glb_save_name = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' +domain+ '/glb_idx_dict_recomb.p'
train_save_name = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' +domain+ '/train_dict_recomb.p'
test_save_name = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' +domain+ '/test_dict_recomb.p'
cPickle.dump(global_idx_dict, open(glb_save_name, 'wb'))
cPickle.dump(train_list, open(train_save_name, 'wb'))
cPickle.dump(test_list, open(test_save_name, 'wb'))

