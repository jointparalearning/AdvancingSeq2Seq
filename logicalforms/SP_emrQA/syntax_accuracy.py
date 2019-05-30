#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 03:41:18 2019

@author: TiffMin
"""

import numpy as np
import _pickle as cPickle
import nltk
import argparse
import copy
from nltk.translate.bleu_score import sentence_bleu


parser = argparse.ArgumentParser(description='Define trainig arguments')
parser.add_argument('-load_dir', '--loading_dir', type=str, metavar='', required=True, help="load model and start with it if there is a directory")
parser.add_argument('-sh', '--shuffle_scheme', type=int, metavar='', required=True, help="shuffle type among 0,1,2")
parser.add_argument('-spl', '--split_num', type=int, metavar='', required=True, help="split among 1,2,3,4,5")
parser.add_argument('-c', '--cuda', type=int, metavar='', required=False, help="split among 1,2,3,4,5")


args = parser.parse_args()



def remove_pad(pair):
    pad_count = sum([1 for i in range(len(pair[1])) if pair[1][i] == 0])
    if pad_count ==0:
        return pair
    else:
        return (pair[0][:-pad_count], pair[1][:-pad_count])

def strip_SOS_EOS_if_twice(tokenized):
    new_tokenized = copy.deepcopy(tokenized) 
    if tokenized[0] == 'SOS_token' and tokenized[1] == 'SOS_token':
        new_tokenized.pop(0)
    if tokenized[-1] == 'EOS_token' and tokenized[-2] == 'EOS_token':
        new_tokenized.pop(-1)
    return new_tokenized
    

def clean_pairs(pairs_dict):
    new_dict = {}
    for k, batch_pairs_list in pairs_dict.items():
        pairs_list_list = []
        for batch_pairs in batch_pairs_list:
            batch_list0 = batch_pairs[0]
            batch_list1 = batch_pairs[1]
            for i, pairs_list in enumerate(batch_list0):
                tup = (batch_list0[i], batch_list1[i])
                #print("before", tup)
                tup = remove_pad(tup)
                #print("removed", tup)
                pairs_list_list.append((list(strip_SOS_EOS_if_twice(tup[0])),list(strip_SOS_EOS_if_twice(tup[1]))))
        new_dict[k] = pairs_list_list
    return new_dict



def main(args):
    def simulate_sorted_idxes(test_batch_num):
        if (test_batch_num+1)*batch_size <= len(validation):
            end_num = (test_batch_num+1)*batch_size 
        else:
            end_num = len(validation)
            #batch_size = end_num - batch_num*batch_size
        sorted_idxes = sorted(validation[test_batch_num*batch_size:end_num], key = lambda idx: len(sent2idxtensor(tokenized_eng_sentences[idx], idx)), reverse=True)
        return sorted_idxes

    def syntax_bleu_acc(pairs_dict, sorted_idexes_dict):
        acc_list = []
        bleu_list = []
        for k, pairs_list in pairs_dict.items():
            acc = 0
            for idx, tup in enumerate(pairs_list):
                tp1, tp2 = tup[0], tup[1]
                idx_of_binary = sorted_idexes_dict[k][idx]
                assert len(tp1) == len(tp2), k
                assert len([0] + lf_binary_entsRAW[idx_of_binary] + [0]) == len(tp1), "tp1: "+str(tp1)+" , " + "binary : " + str(lf_binary_entsRAW[idx_of_binary])
                np_binary = -(np.array([0] + lf_binary_entsRAW[idx_of_binary] + [0]) -1)
                tp1, tp2 = np.array(tp1) * np_binary, np.array(tp2) * np_binary
                acc += list(tp1)==list(tp2)
                bleu = sentence_bleu([list(tp2)], tp1)
                bleu_list.append(bleu)
            acc = acc/len(pairs_list) 
            acc_list.append(acc)
        return acc_list, bleu_list
    
    
    
    global split_num
    global shuffle_scheme 
    
    lf_binary_entsRAW = cPickle.load(open("data/raw_lf_binary_ent.p", "rb"))
    
    split_num = args.split_num
    shuffle_scheme = args.shuffle_scheme
    batch_size = 32
    exec(open('data_prep/data_prepRAW_Shuffle.py').read(),  globals(), globals())
    
    sorted_idexes_dict = {}
    test_batch_num = 0
    while (test_batch_num) * batch_size < len(validation):
        sorted_idexes_dict[test_batch_num+1] = simulate_sorted_idxes(test_batch_num)
        test_batch_num +=1
        batch_size=32
    
    directory = "outputs/" + args.loading_dir +  "/validation_results"
    file_name = directory + "/validation_result.p"
    dict_pairs = cPickle.load(open(file_name, "rb"))
    try:
        tr_pairs = dict_pairs['translation_pairs']
    except:
        tr_pairs = dict_pairs['pairs_dict']['translation_pairs']
    tr_pairs = clean_pairs(tr_pairs)
    syntax_acc_list = syntax_bleu_acc(tr_pairs, sorted_idexes_dict)
    print("syntax acc is : ",  np.mean(syntax_acc_list[0]))
    print("bleu mean is : ",  np.mean(syntax_acc_list[1]))
    cPickle.dump(syntax_acc_list[1], open( directory + "/bleu_list.p", "wb"))
    

###Do BLEU here 

if __name__ == '__main__':
    main(args)
