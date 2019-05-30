#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:20:39 2019

@author: TiffMin
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


import _pickle as cPickle
import numpy as np
import argparse
from torch.nn.functional import cosine_similarity
from sklearn.manifold import TSNE
import copy
import heapq
import torch
import time
import os 
import random




parser = argparse.ArgumentParser(description='Define trainig arguments')
parser.add_argument('-sh', '--shuffle_scheme', type=int, metavar='', required=True, help="shuffle type among 0,1,2")
parser.add_argument('-spl', '--split_num', type=int, metavar='', required=True, help="split among 1,2,3,4,5")
parser.add_argument('-l', '--loading_dir', type=str, metavar='', required=True, help="load model and start with it if there is a directory")
parser.add_argument('-knn_k', '--knn_k', type=int, metavar='', default=10, help="load model and start with it if there is a directory")
parser.add_argument('-tsne_k', '--tsne_k', type=int, metavar='', default=2, help="load model and start with it if there is a directory")
parser.add_argument('-tsne_perp', '--tsne_perp', type=int, metavar='', default=50, help="load model and start with it if there is a directory")


parser.add_argument('-sorted_index', '--sorted_index', type = int, metavar='', required=True)
parser.add_argument('-avg_cos', '--avg_cos', type = int, metavar='', required=True)
parser.add_argument('-knn', '--k_nearest_neighbor', type=int, metavar='', required=True, help="load model and start with it if there is a directory")
parser.add_argument('-tsne', '--tsne', type=int, metavar='', required=True, help="load model and start with it if there is a directory")
parser.add_argument('-c', '--cuda', type=int, metavar='', required=False, help="split among 1,2,3,4,5")
parser.add_argument('-tsne_plot_only', '--tsne_plot_only', type=int, metavar='', default=0, help="load model and start with it if there is a directory")
parser.add_argument('-tsne_plot_lf', '--tsne_plot_lf', type=int, metavar='', default=0, help="load model and start with it if there is a directory")


args = parser.parse_args()


def syntax_bleu_acc(pairs_dict, sorted_idxes_dict):
        acc_list = []
        bleu_list = []
        for k, pairs_list in pairs_dict.items():
            acc = 0
            for idx, tup in enumerate(pairs_list):
                tp1, tp2 = tup[0], tup[1]
                idx_of_binary = sorted_idxes_dict[k][idx]
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



def cos_sim(args):
    
    load_dir = "outputs/" + args.loading_dir + "/validation_results"
    
    cos_sim_inside_same_lf = cPickle.load(open(load_dir + '/cos_sim_inside_same_lf.p','rb'))
    cos_sim_diff_tq_same_lf = cPickle.load(open(load_dir + '/cos_sim_diff_tq_same_lf.p','rb'))
    cos_sim_inside_same_templq = cPickle.load(open(load_dir + '/cos_sim_inside_same_templq.p','rb'))
    cos_sim_for_lf_pairs = cPickle.load(open(load_dir + '/cos_sim_for_lf_pairs.p','rb'))
    
    
    same_lf_cos_avg = np.mean([v for lf, v in cos_sim_inside_same_lf.items() if not(v is None)])
    diff_tq_same_lf_cos_avg = np.mean([v for lf, v in cos_sim_diff_tq_same_lf.items() if not(v is None)])
    same_tq_cos_avg = np.mean([v for lf, v in cos_sim_inside_same_templq.items() if not(v is None)])
    distinct_lf_cos_avg = np.mean([v for lf_pair, v in cos_sim_for_lf_pairs.items()])
    
    print(args.loading_dir, " same_lf_cos_avg: ", same_lf_cos_avg)
    print(args.loading_dir, " diff_tq_same_lf_cos_avg: ", diff_tq_same_lf_cos_avg)
    print(args.loading_dir, " same_tq_cos_avg: ", same_tq_cos_avg)
    print(args.loading_dir, " distinct_lf_cos_avg: ", distinct_lf_cos_avg)


def main(args):
    #cos similarity among same ent and same lf : this needs to be as close as possible, 거의 1 
    #same tq, diff ent보다 가까워야 하는지는 잘 모르겠음 
    global shuffle_scheme
    global split_num
    
    shuffle_scheme = args.shuffle_scheme ; split_num = args.split_num
    
    exec(open('data_prep/data_prepRAW_Shuffle.py').read(), globals(), globals()) 

    load_dir = "outputs/" + args.loading_dir + "/validation_results"
    
    
    #Entries of these are (sum,p_s so far)
    if args.sorted_index == 1:
        sorted_idxes_dict(split_num, shuffle_scheme)
    q_to_p_dict = cPickle.load(open('data/q_to_p_dict.p','rb'))
    hidden_ec_list = cPickle.load(open(load_dir+"/hidden_ec_list.p","rb"))
    if args.tsne == 1:
        tsne(args, rev_unique_templ_q_dict, hidden_ec_list, load_dir)
    if args.avg_cos == 1:
        cos_w_entity(args, hidden_ec_list, rev_unique_templ_q_dict, q_to_p_dict)
    if args.tsne_plot_only ==1:
        X_tsne = cPickle.load(open(load_dir +'/X_tsne_ncomp=' +str(args.tsne_k) + '.p','rb'))
        tsne_plot_only(hidden_ec_list, X_tsne, load_dir)
    


def sorted_idxes_dict(spl, shu):
    def prepare_batch_test(test_batch_num):
        if (test_batch_num+1)*batch_size <= len(validation):
            end_num = (test_batch_num+1)*batch_size 
        else:
            end_num = len(validation)
            #batch_size = end_num - batch_num*batch_size
        sorted_idxes = sorted(validation[test_batch_num*batch_size:end_num], key = lambda idx: len(sent2idxtensor(tokenized_eng_sentences[idx], idx)), reverse=True)
        #print("sorted idxes:" + str(sorted_idxes))
        max_length = len(sent2idxtensor(tokenized_eng_sentences[sorted_idxes[0]], sorted_idxes[0]))
        max_y_length = max([len(Qidx2LFIdxVec_dict[idx]) for idx in sorted_idxes])
        sentence_vectors,binary_vectors, x_lengths, labels = [],[],[],[]
        for idx in sorted_idxes:
            sent = sent2idxtensor(tokenized_eng_sentences[idx], idx)
            sentence_vectors.append(sent +[pad_token] * (max_length-len(sent)))
            x_lengths.append(len(sent))
            labels.append(Qidx2LFIdxVec_dict[idx] +[pad_token] * (max_y_length - len(Qidx2LFIdxVec_dict[idx])) )
        return torch.tensor(sentence_vectors, device=torch.device('cpu')), torch.tensor(binary_vectors, dtype=torch.float, device=torch.device('cpu')), torch.tensor(labels, device=torch.device('cpu')), x_lengths, sorted_idxes
    
    sorted_idxes_dict = {}
    test_batch_num = 0
    batch_size = 32
    while (test_batch_num) * batch_size < len(validation):
        sentence_vectors,binary_vectors, target, X_lengths, sorted_idxes  = prepare_batch_test(test_batch_num)
        test_batch_num +=1
        sorted_idxes_dict[test_batch_num] =sorted_idxes
    sorted_idx_load_dir = 'outputs/sorted_idxes/shuffle' + str(shu) + '/spl' + str(spl)
    if not os.path.exists(sorted_idx_load_dir):
        os.makedirs(sorted_idx_load_dir)
    cPickle.dump(sorted_idxes_dict, open(sorted_idx_load_dir+"/sorted_idxes_dict.p", "wb"))
    


    
    
def cos_w_entity(args, hidden_ec_list, rev_unique_templ_q_dict, q_to_p_dict):
    #cos_tempq = {tq: 0 for tq in unique_templ_q_dict}
    #counter_tempq = {tq: 0 for tq in unique_templ_q_dict}
    cos_lf = {lf: 0 for lf in unique_lf_dict}
    counter_lf = {lf: 0 for lf in unique_lf_dict}
    
    
    for qidx in hidden_ec_list:
        #temp_q = rev_unique_templ_q_dict[qidx]
        lf = rev_unique_lf_dict[qidx]
        ps = q_to_p_dict[qidx]
        emb_q = hidden_ec_list[qidx]
        for p in ps:
            if p in hidden_ec_list:
                emb_p = hidden_ec_list[p]
                cos_sim = cosine_similarity(emb_p,emb_q,0).item()
                if cos_sim >1.0:
                    cos_sim = 1.0
                cos_lf[lf] += cos_sim
                counter_lf[lf] +=1
        
    cos_avg_lf = {}
    total_avg = 0
    total_v = 0
    for lf, v in counter_lf.items():
        if v !=0:
            cos_avg_lf[lf] = cos_lf[lf]/v
            total_avg += cos_lf[lf]
            total_v += v
        else:
            pass
    
    #Negative Examples
    cos_neg_lf = {lf: 0 for lf in unique_lf_dict}
    counter_neg_lf = {lf: 0 for lf in unique_lf_dict}
    for qidx in hidden_ec_list:
        lf = rev_unique_lf_dict[qidx]
        ps = q_to_p_dict[qidx]
        ps_dict = {p: 1 for p in ps}
        emb_q = hidden_ec_list[qidx]
        #sample 5 non-paraphrase 
        complete_non_paras = [p for p in hidden_ec_list if not(p in ps_dict or p==qidx)]
        non_paras = random.sample(complete_non_paras, 10)
        for np in non_paras:
            emb_np = hidden_ec_list[np]
            cos_sim = cosine_similarity(emb_np,emb_q,0).item()
            if cos_sim >1.0:
                cos_sim = 1.0
            cos_neg_lf[lf] += cos_sim
            counter_neg_lf[lf] +=1
    
    cos_neg_avg_lf = {}
    total_neg_avg = 0
    total_neg_v = 0
    for lf, v in counter_neg_lf.items():
        if v !=0:
            cos_neg_avg_lf[lf] = cos_neg_lf[lf]/v
            total_neg_avg += cos_neg_lf[lf]
            total_neg_v += v
        else:
            pass
            
        
    
    print("Avg cosine distance between paraphrases in load dir: " + str(args.loading_dir) + " IS " +str(total_avg/total_v) )
    print("Avg cosine distance between NON-paraphrases in load dir: " + str(args.loading_dir) + " IS " +str(total_neg_avg/total_neg_v) )
    print("Avg Difference: " + str(total_avg/total_v - total_neg_avg/total_neg_v))

    print("All cosine distances between Paraphrases: ")
    print(cos_avg_lf)
    print("All cosine distances between NON-Paraphrases: ")
    print(cos_neg_avg_lf)
    #cPickle.dump(open())
    #cPickle.dump(open())
        
    
        
    
           
def tsne(args,rev_unique_templ_q_dict, hidden_ec_list, load_dir):
    
    def tsned_cosine(X_tsne, Q):
        
        full_dict = {}
        start_start = time.time()
        m = X_tsne.shape[0]
        for i in range(m):
            start = time.time()
            q = Q[i]
            q_emb = torch.tensor(X_tsne[i] , device=torch.device('cpu'))
            full_dict[q] = {}
            for j in range(m):
                p = Q[j]
                p_emb =  torch.tensor(X_tsne[j], device=torch.device('cpu'))
                full_dict[q][p] = cosine_similarity(q_emb,p_emb,0).item()
            if i ==0:
                print("Takes " + str(time.time() - start) + "secs for eqch q in full_dict")
            
        print("Takes " + str(time.time() - start_start) + "secs for the entire full_dict")
        return full_dict
    
    def tsned_k_nearest(number_k, full_dict):
        knn_dict = {}
        start_start = time.time()
        for q in full_dict:
            start = time.time()
            print("current q: ", q)
            d = full_dict[q]
            sort_time = time.time()
            maximums = {k: d[k] for k in heapq.nlargest(number_k, d, key=lambda k: d[k])}
            print("sort time is: ", )
            knn_dict[q] = maximums
            if i ==0:
                print("Takes " + str(time.time() - start) + "secs for sorting each q")
                
        print("Takes " + str(time.time() - start_start) + "secs for the entire knn")
        return knn_dict
    
    #default is 2 for args.n_comp
    tsne = TSNE(n_components=args.tsne_k, init='pca',perplexity = args.tsne_perp) 
    first_key = list(hidden_ec_list.keys())[0]
    m, n = len(hidden_ec_list.keys()), hidden_ec_list[first_key].shape[0]
    assert len(hidden_ec_list[first_key].shape) == 1
    X = np.zeros((m,n))
    Y = np.zeros(m)
    Q = np.zeros(m)
    counter = 0
    for q, v in hidden_ec_list.items():
        X[counter] = v.cpu().numpy()
        temp_q = rev_unique_templ_q_dict[q]
        if args.tsne_plot_lf ==0:
            Y[counter] = temp_q
        else:
            Y[counter] = rev_unique_lf_dict[q]
        Q[counter] = q
        counter +=1
    print("Beginning tsne..")
    start = time.time()
    X_tsne = tsne.fit_transform(X)
    cPickle.dump(X_tsne, open(load_dir + '/X_tsne_ncomp=' +str(args.tsne_k) + '_perp=' + str(args.tsne_perp) + '.p','wb'))
    print("Tsne complete!, time took: " + str(time.time() - start) + "secs" )
    vis_x1 = X_tsne[:,0]; vis_x2 = X_tsne[:,1]
    plt.scatter(vis_x1, vis_x2, c=Y, cmap=plt.cm.get_cmap("jet", 10))
    if args.tsne_plot_lf == 1:
        label_type = 'lf'
    else:
        label_type = 'tempq'
    plt.savefig(load_dir +  '/tsne_perp=' + str(args.tsne_perp) +  '_label=' + str(label_type) + '_scatter.png')    
    print("Saved Plot!")
               
    if args.k_nearest_neighbor ==1:
        print("Beginning k=" + str(args.knn_k)+ "nearest neighbor..")
        start = time.time()
        full_dict = tsned_cosine(X_tsne, Q)
        knn_dict = tsned_k_nearest(args.knn_k, full_dict)
        print("knn completed, time took: " + str(time.time() - start) + "secs" )
        print("saving..")
        cPickle.dump(full_dict, open(load_dir + '/full_dict_cosine_distance_tsne_ncomp=' + str(args.tsne_k) + '.p','wb'))
        cPickle.dump(knn_dict, open(load_dir + '/' + str(args.knn_k) +'_nearest_cosine_distance_tsne_ncomp=' + str(args.tsne_k) + '.p','wb'))


def tsne_plot_only(hidden_ec_list, X_tsne, load_dir):
    m, n = X_tsne.shape
    Y = np.zeros(m)
    counter = 0
    for q, v in hidden_ec_list.items():
        temp_q = rev_unique_templ_q_dict[q]
        if args.tsne_plot_lf ==0:
            Y[counter] = temp_q
        else:
            Y[counter] = rev_unique_lf_dict[q]
        counter +=1
    vis_x1 = X_tsne[:,0]; vis_x2 = X_tsne[:,1]
    plt.scatter(vis_x1, vis_x2, c=Y, cmap=plt.cm.get_cmap("jet", 10))
    if args.tsne_plot_lf == 1:
        label_type = 'lf'
    else:
        label_type = 'tempq'
    plt.savefig(load_dir +  '/tsne_perp=' + str(args.tsne_perp) +  '_label=' + str(label_type) + '_scatter.png')
    print("Saved Plot!")

    #cos similarity among 

if __name__ == '__main__':
    torch.cuda.set_device(args.cuda)
    main(args)


