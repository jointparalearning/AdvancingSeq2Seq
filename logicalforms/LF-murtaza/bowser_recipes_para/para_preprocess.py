#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 01:00:19 2019

@author: TiffMin
"""

import _pickle as cPickle 

#domain = 'blocks'
domain = 'calendar'
#domain = 'basketball'
#domain = 'recipes'
#domain = 'housing'
#domain = 'publications'
#domain = 'socialnetwork'
#domain = 'restaurants'

#file_name = '/Volumes/Transcend/overnightData/recipes.paraphrases.groups.txt'
#file_name = '/Volumes/Transcend/overnightData/blocks.paraphrases.groups.txt'
#file_name = '/Volumes/Transcend/overnightData/calendar.paraphrases.groups.txt'
file_name = '/Volumes/Transcend/overnightData/' + domain + '.paraphrases.groups.txt'


line_list = []
with open(file_name) as open_f:
    for i, line in enumerate(open_f):
        line_list.append(line)
                 
# =============================================================================
# if domain == 'calendar':
#     plus_file_name = '/Volumes/Transcend/overnightData/calendarplus.paraphrases.groups.txt'
#     with open(plus_file_name) as open_f:
#         for i, line in enumerate(open_f):
#             line_list.append(line)
#         
# =============================================================================
#Make dictionary of paraphrases 
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

#Make train_en, train_dcs, test_en, test_dcs instead of reading from them 

train_file_name = '/Volumes/Transcend/overnightData/' + domain + '.paraphrases.train.examples.txt'
test_file_name = '/Volumes/Transcend/overnightData/' + domain +'.paraphrases.test.examples.txt'
# =============================================================================
# if domain == 'calendar':
#     plus_train_file_name = '/Volumes/Transcend/overnightData/calendarplus.paraphrases.train.examples.txt'
#     plus_test_file_name = '/Volumes/Transcend/overnightData/calendarplus.paraphrases.test.examples.txt'
# 
# =============================================================================
train_line_list = {}; test_line_list = {}
train_counter = -1; test_counter = -1
with open(train_file_name) as open_f:
    for i, line in enumerate(open_f):
        if line == '(example\n':
            train_counter +=1
            train_line_list[train_counter] = {'utterance': 0,'para_group': 0, 'answer': 0, 'glb_idx':-1}
            
        if '  (utterance' in line[:20]:
            train_line_list[train_counter]['utterance'] = line[14:-3]
            glb_idx = proc_utt_dict[''.join(train_line_list[train_counter]['utterance'].split(' '))]
            train_line_list[train_counter]['glb_idx'] = glb_idx
            train_line_list[train_counter]['para_group'] = global_idx_dict['para_group'][glb_idx]
            
        elif '    (call' in line[:20]:
            #train_line_list[train_counter]['answer'] = line.replace('call edu.stanford.nlp.sempre.overnight.SimpleWorld.', '').replace('.', ' ').replace('_', ' ')
            train_line_list[train_counter]['answer'] = line.replace('edu.stanford.nlp.sempre.overnight.SimpleWorld.', ' ').replace('en.', '')#.replace('.', ' ').replace('_', ' ')

# =============================================================================
# if domain == 'calendar':  
#     with open(plus_train_file_name) as open_f:
#         for i, line in enumerate(open_f):
#             if line == '(example\n':
#                 train_counter +=1
#                 train_line_list[train_counter] = {'utterance': 0,'para_group': 0, 'answer': 0, 'glb_idx':-1}
#                 
#             if '  (utterance' in line[:20]:
#                 train_line_list[train_counter]['utterance'] = line[14:-3]
#                 glb_idx = proc_utt_dict[''.join(train_line_list[train_counter]['utterance'].split(' '))]
#                 train_line_list[train_counter]['glb_idx'] = glb_idx
#                 train_line_list[train_counter]['para_group'] = global_idx_dict['para_group'][glb_idx]
#                 
#             elif '    (call' in line[:20]:
#                 train_line_list[train_counter]['answer'] = line.replace('edu.stanford.nlp.sempre.overnight.SimpleWorld.', ' ').replace('en.', '')#.replace('call ', '')
# 
#     
# =============================================================================
        
with open(test_file_name) as open_f:
    for i, line in enumerate(open_f):
        if line == '(example\n':
            test_counter +=1
            test_line_list[test_counter] = {'utterance': 0,'para_group': 0, 'answer': 0, 'glb_idx':-1}
        if '  (utterance' in line[:20]:
            test_line_list[test_counter]['utterance'] = line[14:-3]
            glb_idx = proc_utt_dict[''.join(test_line_list[test_counter]['utterance'].split(' '))]
            test_line_list[test_counter]['glb_idx'] = glb_idx
            test_line_list[test_counter]['para_group'] = global_idx_dict['para_group'][glb_idx]
            
        elif '    (call' in line[:20]:
            #test_line_list[test_counter]['answer'] = line.replace('call edu.stanford.nlp.sempre.overnight.SimpleWorld.', '').replace('.', ' ').replace('_', ' ')
            test_line_list[test_counter]['answer'] = line.replace('edu.stanford.nlp.sempre.overnight.SimpleWorld.', ' ').replace('en.', '')#.replace('.', ' ').replace('_', ' ')
# =============================================================================
# if domain == 'calendar':  
#     with open(plus_test_file_name) as open_f:
#         for i, line in enumerate(open_f):
#             if line == '(example\n':
#                 test_counter +=1
#                 test_line_list[test_counter] = {'utterance': 0,'para_group': 0, 'answer': 0, 'glb_idx':-1}
#                 
#             if '  (utterance' in line[:20]:
#                 test_line_list[test_counter]['utterance'] = line[14:-3]
#                 glb_idx = proc_utt_dict[''.join(test_line_list[test_counter]['utterance'].split(' '))]
#                 test_line_list[test_counter]['glb_idx'] = glb_idx
#                 test_line_list[test_counter]['para_group'] = global_idx_dict['para_group'][glb_idx]
#                 
#             elif '    (call' in line[:20]:
#                 test_line_list[test_counter]['answer'] = line.replace('edu.stanford.nlp.sempre.overnight.SimpleWorld.', ' ').replace('en.', '')#.replace('call ', '')
# 
# =============================================================================

glb_save_name = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' +domain+ '/glb_idx_dict.p'
train_save_name = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' +domain+ '/train_dict.p'
test_save_name = '/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' +domain+ '/test_dict.p'
cPickle.dump(global_idx_dict, open(glb_save_name, 'wb'))
cPickle.dump(train_line_list, open(train_save_name, 'wb'))
cPickle.dump(test_line_list, open(test_save_name, 'wb'))


# =============================================================================
# #existing train test split   
# #동시에 global dict of lf's 
# with open(train_en) as open_f:
#     train_sents = []
#     for line in open_f:
#         train_sents.append(line.strip())
#         
# with open(train_dcs) as open_f:
#     train_lfs = []
#     for line in open_f:
#         train_lfs.append(line.strip())
# 
# already_overlap = [] 
# for i, train_sent in enumerate(train_sents):
#     #assert ''.join(train_sent.split(' ')) in proc_utt_dict, i 
#     proc_train_sent = ''.join(train_sent.split(' '))
#     gold_train_2_global_idx_dict[i] = proc_utt_dict[proc_train_sent]
#     global_idx_2_gold_train_idx_dict[proc_utt_dict[proc_train_sent]] = i
#     #if proc_utt_dict[proc_test_sent] in para_lf_idx_dict:
#     #    already_overlap.append(i)
#     para_lf_idx_dict[proc_utt_dict[proc_train_sent]] = train_lfs[i]
#   
# with open(test_en) as open_f:
#     test_sents = []
#     for line in open_f:
#         test_sents.append(line.strip())
# 
# with open(test_dcs) as open_f:
#     test_lfs = []
#     for line in open_f:
#         test_lfs.append(line.strip())
# 
# overlap_in_training = []
# for i, test_sent in enumerate(test_sents):
#     #assert ''.join(train_sent.split(' ')) in proc_utt_dict, i 
#     proc_test_sent = ''.join(test_sent.split(' '))
#     gold_test_2_global_idx_dict[i] = proc_utt_dict[proc_test_sent]
#     global_idx_2_gold_test_idx_dict[proc_utt_dict[proc_test_sent]] = i
#     #if proc_utt_dict[proc_test_sent] in para_lf_idx_dict:
#     #    overlap_in_training.append(i)
#     para_lf_idx_dict[proc_utt_dict[proc_test_sent]] = test_lfs[i]
#   
# 
# global_idx_dict['lf'] = para_lf_idx_dict 
# 
# #How to Test 
# 
# weird = []
# for k, v in para_utterance_idx_dict.items():
#     if not(k in para_lf_idx_dict):
#         weird.append(k)
#     
# #Assign paraphrase group to each index in training 
# train_para_group ={}
# for i, train_sent in enumerate(train_sents):
#     proc_train_sent = ''.join(train_sent.split(' '))
#     global_idx = proc_utt_dict[proc_train_sent]
#     train_para_group[i] = global_idx_dict['para_group'][global_idx]
#     
# 
# 
# for i in range(len(global_idx_dict['utterance'])):
#     if not(i in global_idx_dict['lf']):
#         para_group = global_idx_dict['para_group'][i]
#         utt = global_idx_dict['utterance'][i]
#         for glb_idx in para_group_dict[para_group]:
#             if glb_idx in global_idx_dict['lf'] and global_idx_dict['utterance'][i] == utt:
#                 global_idx_dict['lf'][i] = global_idx_dict['lf'][glb_idx]
#         
# =============================================================================

#Save to Pickle  - recipes original
# =============================================================================
# cPickle.dump(global_idx_dict, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/global_idx_dict.p','wb'))
# cPickle.dump(train_para_group, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/shuffle0_train2_para_group_dict.p','wb'))
# cPickle.dump(para_group_dict, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/para_group_dict.p','wb'))
# cPickle.dump(para_utterance_idx_dict, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/global_idx2_utterance_dict.p', 'wb')) 
#  
# cPickle.dump(gold_train_2_global_idx_dict, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/shuffle0_train_2_global_idx_dict.p','wb')) 
# cPickle.dump(gold_test_2_global_idx_dict, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/shuffle0_test_2_global_idx_dict.p','wb')) 
# 
# cPickle.dump(global_idx_2_gold_test_idx_dict, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/global_idx_2_shuffle0_test_idx_dict.p','wb'))
# cPickle.dump(global_idx_2_gold_train_idx_dict, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/global_idx_2_shuffle0_train_idx_dict.p','wb'))
# 
# =============================================================================
#
                
#Save to Pickle - blocks 
# =============================================================================
# cPickle.dump(global_idx_dict, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain +'/global_idx_dict.p','wb'))
# cPickle.dump(train_para_group, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' +domain+'/shuffle0_train2_para_group_dict.p','wb'))
# cPickle.dump(para_group_dict, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain +'/para_group_dict.p','wb'))
# cPickle.dump(para_utterance_idx_dict, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' +domain + '/global_idx2_utterance_dict.p', 'wb')) 
#  
# cPickle.dump(gold_train_2_global_idx_dict, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' +domain+ '/shuffle0_train_2_global_idx_dict.p','wb')) 
# cPickle.dump(gold_test_2_global_idx_dict, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' +domain+ '/shuffle0_test_2_global_idx_dict.p','wb')) 
# 
# cPickle.dump(global_idx_2_gold_test_idx_dict, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' + domain +'/global_idx_2_shuffle0_test_idx_dict.p','wb'))
# cPickle.dump(global_idx_2_gold_train_idx_dict, open('/Users/TiffMin/Desktop/Tiffany/github/logicalforms/LF-murtaza/bowser_recipes_para/data/para_preprocessed/' +domain+ '/global_idx_2_shuffle0_train_idx_dict.p','wb'))
# 
# =============================================================================
