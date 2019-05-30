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
import pickle, copy
import _pickle as cPickle

#Ignore Argparse
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

def acc(pairs_dict):
    acc_list = []
    for k, pairs_list in pairs_dict.items():
        if len(pairs_list)>0:
            acc = sum([list(tup[0]) == list(tup[1]) for tup in pairs_list])/len(pairs_list)
        else:
            acc=0
        acc_list.append(acc)
    return acc_list

def syntax_acc(pairs_dict, sorted_idexes_dict):
    acc_list = []
    for k, pairs_list in pairs_dict.items():
        for idx, tup in enumerate(pairs_list):
            tp1, tp2 = tup[0], tup[1]
            idx_of_binary = sorted_idexes_dict[k][idx]
            assert len(tp1) == len(tp2), k
            assert len([0] + lf_binary_entsRAW[idx_of_binary] + [0]) == len(tp1)
            np_binary = -(np.array([0] + lf_binary_entsRAW[idx_of_binary] + [0]) -1)
            tp1, tp2 = np.array(tp1) * np_binary, np.array(tp2) * np_binary
        acc = sum([list(tup[0]) == list(tup[1]) for tup in pairs_list])/len(pairs_list)
        acc_list.append(acc)
    return acc_list
    


def acc_for_geo(tr_pairs):
    acc_list = []
    for k,v in tr_pairs.items():
        acc = 0
        for i in range(len(v)):
            acc += tr_pairs[k][i][0] == tr_pairs[k][i][1]
        acc_list.append(acc / len(v))
    return acc_list



##
## Shows and saves plot from pickled predicted, target. 
## file_name ('str'): name of the pickle file that contains predicted, target to read from 
## save_name ('str'): name of the graph.png file to save 
## save (True/False): whether to save file with the save_name
## show (True/False): whether to show graph in the ipython console
##

def plot(file_name, save_name, save=True, show=True):
    dict_pairs = cPickle.load(open(file_name, "rb"))
    try:
        tr_pairs = dict_pairs['translation_pairs']
    except:
        tr_pairs = dict_pairs['pairs_dict']['translation_pairs']
    tr_pairs = clean_pairs(tr_pairs)
    acc_list = acc(tr_pairs)
    plt.plot(acc_list)
    plt.title(save_name)
    if save:
        plt.savefig(save_name +'.png')
    if show:
        plt.show()
    plt.close()
 
##
## Shows and saves plot from two pickles at the same plot. Label them with correct labels. 
## file_name1 ('str'): name of the first pickle file that contains predicted, target to read from 
## file_name2 ('str'): name of the second pickle file that contains predicted, target to read from
## label1 ('str'): label for the first file, for example, "alpha = 0.0"
## label2 ('str'): label for the 2nd file, for example, "alpha = 1.0" 
## save_name ('str'): name of the graph.png file to save 
## save (True/False): whether to save file with the save_name
## show (True/False): whether to show graph in the ipython console
##

def double_plot(file_name1, file_name2, lab1, lab2,file_name3 = None, file_name4 = None, lab3 = None, lab4 = None, save_name='acc', save=True, show=True):
    dict_pairs1 = cPickle.load(open(file_name1, "rb"))
    dict_pairs2 = cPickle.load(open(file_name2, "rb"))
    if not(file_name3 is None):
        dict_pairs3 = cPickle.load(open(file_name3, "rb"))
    if not(file_name4 is None):
        dict_pairs4 = cPickle.load(open(file_name4, "rb"))
    try:
        tr_pairs1 = dict_pairs1['translation_pairs']
        tr_pairs2 = dict_pairs2['translation_pairs']
        if not(file_name3 is None):
            tr_pairs3 = dict_pairs3['translation_pairs']
        if not(file_name4 is None):
            tr_pairs4 = dict_pairs4['translation_pairs']

    except:
        tr_pairs1 = dict_pairs1['pairs_dict']['translation_pairs']
        tr_pairs2 = dict_pairs2['pairs_dict']['translation_pairs']
        if not(file_name3 is None):
            tr_pairs3 = dict_pairs3['pairs_dict']['translation_pairs']
        if not(file_name4 is None):
            tr_pairs4 = dict_pairs4['pairs_dict']['translation_pairs']

        
    tr_pairs1 = clean_pairs(tr_pairs1)
    tr_pairs2 = clean_pairs(tr_pairs2)
    if not(file_name3 is None):
        tr_pairs3 = clean_pairs(tr_pairs3)
    if not(file_name4 is None):
        tr_pairs4 = clean_pairs(tr_pairs4)
    
    acc_list1 = acc(tr_pairs1); acc_list2 = acc(tr_pairs2);
    if not(file_name3 is None):
        acc_list3 = acc(tr_pairs3); 
    if not(file_name4 is None):
        acc_list4 = acc(tr_pairs4)

    plt.plot(acc_list1, label=lab1)
    plt.plot(acc_list2, label=lab2)
    try:
        plt.plot(acc_list3, label=lab3)
        plt.plot(acc_list4, label=lab4)
    except:
        pass
    plt.legend()
    
    plt.title(save_name)
    if save:
        plt.savefig(save_name +'.png')
    if show:
        plt.show()
    plt.close()


###Example for plot()
os.chdir("/Users/TiffMin/Desktop/Tiffany/github/logicalforms/Refactored_Tiffany")
directory = 'testcopycon'
file_name = 'outputs/' + directory + '/loss_list.p'
#save_name = 'outputs/' + directory + '/'+ directory+ 'acc.png'
save_name =  directory  
plot(file_name, save_name, save=False, show=True)

directory1 = 'split1/shuffle0/3e-3_no_decay_m1kl0'
directory2 = 'split1/shuffle0/3e-3_no_decay_m0alpha0'
#directory3 = 'split1/shuffle2/3e-3_no_decay_m0alpha0'
#directory4 = 'split1/shuffle1/smalllr_m0alpha1'

lab1 = 'Proposed Model: kl 0, vae'
lab2 = 'alpha 0, plain'
#lab3 = 'alpha 0, plain'
#lab4 = 'alpha 1, plain'

file_name1 = 'outputs/' + directory1 + '/loss_list.p'
file_name2 = 'outputs/' + directory2 + '/loss_list.p'
#file_name3 = 'outputs/' + directory3 + '/loss_list.p'
#file_name4 = 'outputs/' + directory4 + '/loss_list.p'


double_plot(file_name1, file_name2, lab1, lab2, file_name3 = file_name3, lab3 = lab3, save_name = 'shuffle scheme #0, split 1, learning rate 1e-4, weight decay 0.985', save=False, show=True)




###Example for double_plot()
directory1 = 'split1/shuffle2/smalllr_m0alpha1kl0'
directory2 = 'split1/shuffle2/smalllr_m0alpha0'
lab1 = 'alpha 1, vae'
lab2 = 'alpha 0, plain'
file_name1 = 'outputs/' + directory1 + '/loss_list.p'
file_name2 = 'outputs/' + directory2 + '/loss_list.p'
double_plot(file_name1, file_name2, lab1, lab2, save_name = 'learning rate 0.0001, weight decay 985', save=False, show=True)


directory1 = 'split1/shuffle2/3e-4_no_decay_m1kl0'
directory2 = 'split1/shuffle2/3e-4_no_decay_m0alpha1'
lab1 = 'alpha 1, vae'
lab2 = 'alpha 1, plain'
file_name1 = 'outputs/' + directory1 + '/loss_list.p'
file_name2 = 'outputs/' + directory2 + '/loss_list.p'
double_plot(file_name1, file_name2, lab1, lab2, save_name = 'learning rate 0.0001, weight decay 985', save=False, show=True)


directory1 = 'split1/shuffle2/3e-4_no_decay_m0alpha0'
directory2 = 'split1/shuffle2/3e-4_no_decay_m0alpha1'
lab1 = 'alpha 0, plain'
lab2 = 'alpha 1, plain'
file_name1 = 'outputs/' + directory1 + '/loss_list.p'
file_name2 = 'outputs/' + directory2 + '/loss_list.p'
double_plot(file_name1, file_name2, lab1, lab2, save_name = 'learning rate 0.0001, weight decay 985', save=False, show=True)


directory1 = 'split1/shuffle2/smalllr_m0alpha1kl0'
directory2 = 'split1/shuffle2/smalllr_m0alpha1kl0'
lab1 = 'kl 0, vae'
lab2 = 'kl 1, vae'
file_name1 = 'outputs/' + directory1 + '/loss_list.p'
file_name2 = 'outputs/' + directory2 + '/loss_list.p'
double_plot(file_name1, file_name2, lab1, lab2, save_name = 'learning rate 0.0001, weight decay 985', save=False, show=True)

##Ignore Argparse
# =============================================================================
# if __name__ == '__main__':
#     directory = args.directory
#     file_name = 'outputs/' + directory + '/loss_list.p'
#     save_name = 'outputs/' + directory + '/'+ directory+ 'acc.png'
#     plot(file_name, save_name, show=True)
#     
# =============================================================================

directory = 'split1/shuffle2/3e-3_no_decay_m0alpha0'
file_name = 'outputs/' + directory + '/loss_list.p'
dict_pairs = cPickle.load(open(file_name, "rb"))
tr_pairs = dict_pairs['pairs_dict']['translation_pairs']
tr_pairs = clean_pairs(tr_pairs)
acc_list = acc(tr_pairs)
acc_list[-1]
