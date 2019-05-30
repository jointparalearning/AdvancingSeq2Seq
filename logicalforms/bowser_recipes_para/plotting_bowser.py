#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 20:56:05 2019

@author: TiffMin
"""

import argparse


import os
import matplotlib.pyplot as plt
import numpy as np
import copy
import pickle as cPickle
from bowser_constants import constants


# =============================================================================
# parser = argparse.ArgumentParser(description='Define trainig arguments')
# 
# parser.add_argument('-dir', '--directory', type = str, metavar='', required =True, help ="pickle directory")
# #parser.add_argument('f', '--filname', type = str, metavar='', required =True, help ="pickle filename")
# #parser.add_argument('s', '--savenum', type=str, metavar='', required=True, help= "savename")
# args = parser.parse_args()
# 
# =============================================================================
    
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
        # print("Cleaning for epoch:", k)
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

def acc(pairs_dict, printing=False):
    acc_list = []
    for k, pairs_list in pairs_dict.items():
        # print("Doing for epoch:", k)
        if len(pairs_list)>0:
            acc = sum([list(tup[0]) == list(tup[1]) for tup in pairs_list])/len(pairs_list)
        else:
            acc=0
        acc_list.append(acc)
    return acc_list

def acc_tokens(pairs_dict, file_str, printing=True):
    acc_list = []

    f = open("results/plots/output_pairs_%s.txt" % file_str, "w")

    with open("data/processed/vocab_recipes_v1.pkl", "rb") as b:
        vocab = cPickle.load(b)

    i2w = vocab["idx2word"]

    for k, pairs_list in pairs_dict.items():
        if printing:
            print("In epoch %d" % k)
        f.write("\n\nFor epoch: %d"%k)
        # print("Doing for epoch:", k)
        if len(pairs_list)>0:
            correct = 0
            token_count = 0
            for pred, actual in pairs_list:
                # Print out the pairs using the vocab and send to a file
                pred_words = []
                actual_words = []
                token_count += len(pred)
                for i in range(len(pred)):
                    pred_words.append(i2w[pred[i]])
                    actual_words.append(i2w[actual[i]])
                    if pred[i]==actual[i]:
                        correct+=1
                f.write("\n"+str(pred_words))
                f.write("\n"+str(actual_words)+"\n")
                if str(pred_words) == str(actual_words):
                    if printing:
                        print()
                        print(pred_words)
                        print(actual_words)
                        print()
            acc_list.append(correct/token_count)
        else:
            acc=0
            acc_list.append(acc)
    return acc_list

def plot(file_name, save_name, show=True, tokens=False, file_str=""):
    dict_pairs = cPickle.load(open(file_name, "rb"))
    # try:
    tr_pairs = dict_pairs['translation_pairs']
    # except:
    #     tr_pairs = dict_pairs['pairs_dict']['translation_pairs']
    tr_pairs = clean_pairs(tr_pairs)
    if tokens:
        acc_list = acc_tokens(tr_pairs, file_str)
    else:
        acc_list = acc(tr_pairs)
    plt.plot(acc_list)
    plt.title(save_name)
    plt.savefig(save_name +'.png')
    if show:
        plt.show()
    plt.close()
 

def double_plot(file_name1, file_name2, lab1, lab2, save_name='acc', save=True, show=True):
    dict_pairs1 = cPickle.load(open(file_name1, "rb"))
    dict_pairs2 = cPickle.load(open(file_name2, "rb"))
    try:
        tr_pairs1 = dict_pairs1['translation_pairs']
        tr_pairs2 = dict_pairs2['translation_pairs']
    except:
        tr_pairs1 = dict_pairs1['pairs_dict']['translation_pairs']
        tr_pairs2 = dict_pairs2['pairs_dict']['translation_pairs']
        
    tr_pairs1 = clean_pairs(tr_pairs1)
    tr_pairs2 = clean_pairs(tr_pairs2)
    
    acc_list1 = acc(tr_pairs1)
    acc_list2 = acc(tr_pairs2)

    plt.plot(acc_list1, label=lab1)
    plt.plot(acc_list2, label=lab2)
    plt.legend()
    
    plt.title(save_name)
    if save:
        plt.savefig(save_name +'.png')
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    # directory = args.directory
    # file_name = 'outputs/' + directory + '/loss_list.p'
    # save_name = 'outputs/' + directory + '/'+ directory+ 'acc.png'
    # plot(file_name, save_name, show=True)
    #
    # os.chdir("/Users/TiffMin/Desktop/Tiffany/github/logicalforms/Refactored_Tiffany")
    # directory = 'model1alpha1.0shuffle2loss0'
    # file_name = 'outputs/' + directory + '/loss_list.p'
    # # save_name = 'outputs/' + directory + '/'+ directory+ 'acc.png'
    # save_name = 'outputs/' + directory + '/' + 'weight_decay%10'
    # plot(file_name, save_name, show=True)
    #
    # directory1 = 'model0alpha0.0shuffle0'
    # directory2 = 'model1alpha1.0shuffle0loss0'
    # lab1 = 'plain'
    # lab2 = 'vae'
    # file_name1 = 'outputs/' + directory1 + '/loss_list.p'
    # file_name2 = 'outputs/' + directory2 + '/loss_list.p'
    # double_plot(file_name1, file_name2, lab1, lab2, save=False, show=True)

    file_str = "run_1_train_v1_alpha1"

    pkl_dir = constants.pkl_dir
    pkl_file = os.path.join(pkl_dir, "validation_pairs_v1_run1_final")
    plt_dir = os.path.join(constants.results_dir, "plots")
    save_filename = os.path.join(plt_dir, "%s_accuracy" % file_str)
    plot(pkl_file, save_filename, False)
    save_filename = os.path.join(plt_dir, "%s_tokens_accuracy" % file_str)
    plot(pkl_file, save_filename, False, True, file_str)

