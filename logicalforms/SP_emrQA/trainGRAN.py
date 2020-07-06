#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:07:08 2019

@author: TiffMin
"""
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from models.functions import numpy_to_var, toData, to_np, to_var, visualize, decoder_initial, update_logger
import time
import sys, os
import math
from numpy import inf
import random
import _pickle as cPickle
import argparse
from models.plain_method2 import *
from models.seq2seq_Luong import *
from paraphrase_embedding_model import *
import copy, math
#from bert_serving.client import BertClient


#from tensorboardX import SummaryWriter




#from models.plain import 
#from models.vae_ver1 import 
#from models. import 

parser = argparse.ArgumentParser(description='Define trainig arguments')

parser.add_argument('-sh', '--shuffle_scheme', type = int, metavar='', required =True, help ="saving directory of output file")
parser.add_argument('-spl', '--split_num', type=int, metavar='', required=True, help= "split num")
parser.add_argument('-save_dir', '--saving_dir', type=str, metavar='', required=True, help="saving directory of output file; not required but can specify if you want to.")


parser.add_argument('-lam_w', '--lam_w', type = float, metavar='', default = 0)
parser.add_argument('-margin', '--margin', type = float, metavar='', default = 0.4)


##Optional Arguments
parser.add_argument('-c', '--cuda', type = int, metavar='', required=False, help = "cuda")

parser.add_argument('-clip', '--clip', type = float, metavar='', default = 0.0, help = "cuda")

parser.add_argument('-gray', '--gray', type = int, metavar='', default = 0, help = "cuda")


args = parser.parse_args()


gray_file = ''
if args.gray == 1:
    gray_file = '/scratch/symin95/gray_emrqa_outputs/'




def train(num_epochs, args):    
    def save( args):
        torch.save(encoder.state_dict(), gray_file + 'outputs/' + args.saving_dir + '/encoder.pt')
               
    def save_every_epoch(args, epoch):
        torch.save(encoder.state_dict(),gray_file + 'outputs/' + args.saving_dir + '/encoder_epoch' + str(epoch) +  '.pt')
                
    def prepare_batch_training(batch_num):
        if (batch_num+1)*batch_size <= len(training_sampled):
            end_num = (batch_num+1)*batch_size 
        else:
            end_num = len(training_sampled)
            #batch_size = end_num - batch_num*batch_size
        #sort by length
        sorted_idxes = sorted(training_sampled[batch_num*batch_size:end_num], key = lambda idx: len(sent2idxtensor(tokenized_eng_sentences[idx], idx)), reverse=True)
        rel_sorted_idxes = sorted(range(batch_num*batch_size,end_num), key = lambda idx: len(sent2idxtensor(tokenized_eng_sentences[training_sampled[idx]], idx)), reverse=True)
        #print("sorted idxes is " + str(sorted_idxes))
        max_length = len(sent2idxtensor(tokenized_eng_sentences[sorted_idxes[0]], sorted_idxes[0]))
        max_y_length = max([len(Qidx2LFIdxVec_dict[idx]) for idx in sorted_idxes])
        #print("max length is " + str(max_length))
        sentence_vectors, binary_vectors, x_lengths, labels = [],[],[],[]
        for idx in sorted_idxes:
            sent = sent2idxtensor(tokenized_eng_sentences[idx], idx)
            sentence_vectors.append(sent +[pad_token] * (max_length-len(sent)))
            x_lengths.append(len(sent))
            labels.append(Qidx2LFIdxVec_dict[idx] +[pad_token] * (max_y_length - len(Qidx2LFIdxVec_dict[idx])) )
        return torch.tensor(sentence_vectors, device=device), torch.tensor(binary_vectors, dtype=torch.float, device = device), torch.tensor(labels, device=device), x_lengths, sorted_idxes, rel_sorted_idxes
                
    
    def prepare_q_to_p_training(batch_num):
        #TODO HERE
        p_vectors = []
        p_idxes = []
        SOS_token = 1; EOS_token = 2

        #valid_count = 0
        for idx in sorted_idxes:
            if q_to_p[idx] != []:
                valid_ps = [p for p in q_to_p[idx] if p in training_sampled]
                if valid_ps != []:
                    random.seed(batch_num)
                    p_idx = random.sample(valid_ps,1)    
                    #valid_count +=1
                else:
                    p_idx = idx
            else:
                p_idx = idx
            p_idxes.append(p_idx)
        
        max_p_length = 2 + max([len(tokenized_eng_sentences[p_idx[0]]) for p_idx in p_idxes])
        temp_p_vectors = []
        for p_idx in p_idxes:
            p_vector = [SOS_token] + [Qvocab2idx[token] for token in tokenized_eng_sentences[p_idx[0]]]+[EOS_token]
            temp_p_vectors.append(p_vector)
            p_vector = p_vector + [pad_token] * (max_p_length - len(p_vector))
            p_vectors.append(p_vector)
        
        sorted_p_vectors = sorted(temp_p_vectors, key=len, reverse=True)
        original_relative_idx_sorted = sorted(range(len(temp_p_vectors)), key=lambda k: temp_p_vectors[k], reverse=True)
        sorted_back_to_original_dict = {idx:i for i,idx in enumerate(original_relative_idx_sorted)}
        sorted_back_to_original_list = [-1] * len(p_idxes)
        for idx, i in sorted_back_to_original_dict.items():
            sorted_back_to_original_list[idx] = i
        sorted_p_lengths = [len(sorted_l) for sorted_l in sorted_p_vectors]
        return_sorted_p_vectors = [sorted_inp + [pad_token] * (max_p_length - len(sorted_inp))for sorted_inp in sorted_p_vectors]

        
        #print("non-auto percent:" , valid_count/batch_size)
        return  torch.tensor(return_sorted_p_vectors, device = device), sorted_back_to_original_list, sorted_p_lengths
          
    def prepare_nonpara(batch_num):
        #TODO HERE
        np1_idxes = []
        np2_idxes = []
        SOS_token = 1; EOS_token = 2

        #valid_count = 0
        for idx in sorted_idxes:
            succeed = 0
            while succeed==0: 
                sampled = random.sample(training_sampled, 2)
                if sampled[0] in q_to_p[idx] or sampled[1] in q_to_p[idx]:
                    succeed = 0
                else:
                    succeed = 1
            np1_idxes.append(sampled[0]); np2_idxes.append(sampled[1])
        
        
        #np1
        max_p_length = 2 + max([len(tokenized_eng_sentences[p_idx]) for p_idx in np1_idxes])
        temp_p_vectors = []
        p_vectors = []
        for p_idx in np1_idxes:
            p_vector = [SOS_token] + [Qvocab2idx[token] for token in tokenized_eng_sentences[p_idx]]+[EOS_token]
            temp_p_vectors.append(p_vector)
            p_vector = p_vector + [pad_token] * (max_p_length - len(p_vector))
            p_vectors.append(p_vector)
        
        sorted_p_vectors = sorted(temp_p_vectors, key=len, reverse=True)
        original_relative_idx_sorted = sorted(range(len(temp_p_vectors)), key=lambda k: temp_p_vectors[k], reverse=True)
        sorted_back_to_original_dict = {idx:i for i,idx in enumerate(original_relative_idx_sorted)}
        sorted_back_to_original_list = [-1] * len(np1_idxes)
        for idx, i in sorted_back_to_original_dict.items():
            sorted_back_to_original_list[idx] = i
        sorted_p_lengths = [len(sorted_l) for sorted_l in sorted_p_vectors]
        return_sorted_p_vectors = [sorted_inp + [pad_token] * (max_p_length - len(sorted_inp))for sorted_inp in sorted_p_vectors]

        return_sorted_np1_vectors = copy.deepcopy(return_sorted_p_vectors)   
        sorted_np1_lengths = copy.deepcopy(sorted_p_lengths)
        sorted_back_to_original_list_np1 = copy.deepcopy(sorted_back_to_original_list)
        
        #np2
        max_p_length = 2 + max([len(tokenized_eng_sentences[p_idx]) for p_idx in np2_idxes])
        temp_p_vectors = []
        p_vectors = []
        for p_idx in np2_idxes:
            p_vector = [SOS_token] + [Qvocab2idx[token] for token in tokenized_eng_sentences[p_idx]]+[EOS_token]
            temp_p_vectors.append(p_vector)
            p_vector = p_vector + [pad_token] * (max_p_length - len(p_vector))
            p_vectors.append(p_vector)
        
        sorted_p_vectors = sorted(temp_p_vectors, key=len, reverse=True)
        original_relative_idx_sorted = sorted(range(len(temp_p_vectors)), key=lambda k: temp_p_vectors[k], reverse=True)
        sorted_back_to_original_dict = {idx:i for i,idx in enumerate(original_relative_idx_sorted)}
        sorted_back_to_original_list = [-1] * len(np2_idxes)
        for idx, i in sorted_back_to_original_dict.items():
            sorted_back_to_original_list[idx] = i
        sorted_p_lengths = [len(sorted_l) for sorted_l in sorted_p_vectors]
        return_sorted_p_vectors = [sorted_inp + [pad_token] * (max_p_length - len(sorted_inp))for sorted_inp in sorted_p_vectors]
        
        return_sorted_np2_vectors = copy.deepcopy(return_sorted_p_vectors)   
        sorted_np2_lengths = copy.deepcopy(sorted_p_lengths)
        sorted_back_to_original_list_np2 = copy.deepcopy(sorted_back_to_original_list)
        
        
        #print("non-auto percent:" , valid_count/batch_size)
        return  torch.tensor(return_sorted_np1_vectors, device = device), sorted_back_to_original_list_np1, sorted_np1_lengths, torch.tensor(return_sorted_np2_vectors, device = device), sorted_back_to_original_list_np2, sorted_np2_lengths
        
    #writer = SummaryWriter()
    global shuffle_scheme
    global split_num
    shuffle_scheme, save_dir, split_num =  args.shuffle_scheme,  args.saving_dir, args.split_num
        

    ###make folder for save_dir
    save_dir = gray_file + "outputs/" + save_dir 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    ###load the required arguments first
    #load shuff_scheme
    exec(open('data_prep/data_prepRAW_Shuffle.py').read(), globals(), globals())        

    
    ###load optional arguments and model
    #load hyperparameters
    hidden_dim = 128 
    learning_rate = 0.001 
    lam_c = 0; lam_w = args.lam_w; margin = args.margin

    #load model version

    encoder = ParaEncoderGRAN(hidden_dim, vocab_size)
    w_initial = torch.detach(encoder.embedding.weight).cuda()
        

    if torch.cuda.is_available():
        encoder.cuda()

    
    #load save_dir
    
    #load loss function
    temp_coss_loss = torch.nn.CosineEmbeddingLoss()
    def hinge_cos_loss(anchors, positives, negatives, margin):
        total_loss = 0.0
        ones = torch.tensor([1.0]).cuda()
        for i in range(anchors.shape[0]):
            anchor = anchors[i]; positive = positives[i]; negative = negatives[i]
            total_loss += max((temp_coss_loss(anchor.view(1,-1), positive.view(1,-1), ones)-1) - (temp_coss_loss(anchor.view(1,-1), negative.view(1,-1), ones)-1) + margin, torch.tensor(0.0).cuda())
        return total_loss/anchors.shape[0]
 
        #cos_loss_function = hinge_cos_loss
    def total_loss(s_1, p, np1, np2, lam_c, lam_w, margin):
        hinges =  hinge_cos_loss(s_1, p, np1, margin) + hinge_cos_loss(p, s_1, np2, margin) 
        encoder_params = list(encoder.named_parameters())
        lstm_wc = sum([param[1].norm(p=2) for param in encoder_params if param[0]!='embedding.weight'])
        #list(paraencoder.named_parameters())
        word_embedding_diff = (encoder.embedding.weight - w_initial).norm(p=2)
        return hinges + lam_c * lstm_wc + lam_w * word_embedding_diff 
    #load model 
        
    epochs = range(num_epochs)  
    q_to_p = cPickle.load(open('data/q_to_p_dict.p','rb'))
    batch_size = math.ceil(len(training_sampled)/100)
    original_batch_size = batch_size
    
    start = time.time()
    step = 0
    for epoch in epochs:
        
        opt_e = optim.Adam(params=encoder.parameters(),  lr=learning_rate)
                
        batch_num = 0         
        print("==================================================")
        print("Epoch ",epoch)
        while (batch_num) * batch_size < len(training_sampled):
            if batch_num % 100 ==0:
                print("==================================================")
                print("Batch Num: ",batch_num)
                print("Batch Percent: ",100 *(batch_num) * batch_size/ len(training_sampled), "%")
            sentence_vectors, binary_vectors, target, X_lengths, sorted_idxes, rel_sorted_idxes = prepare_batch_training(batch_num)
            sorted_p_vectors, sorted_back_to_original_list, sorted_p_lengths = prepare_q_to_p_training(batch_num)
            sorted_np1_vectors, sorted_back_to_original_list_np1, sorted_np1_lengths,sorted_np2_vectors, sorted_back_to_original_list_np2, sorted_np2_lengths = prepare_nonpara(batch_num)
            
            #batch_size = sentence_vectors.shape[0]
            batch_size = sentence_vectors.shape[0]
            opt_e.zero_grad()
            # apply to encoder
            #Need to put in x, paraphrase, x's non paraphrase
            hidden_ec = encoder(sentence_vectors, X_lengths)[1] # z is [batch_size, latent_size]
            hidden_ec_p = encoder(sorted_p_vectors, sorted_p_lengths)[1]
            hidden_ec_np1 = encoder(sorted_np1_vectors, sorted_np1_lengths)[1]
            hidden_ec_np2 = encoder(sorted_np2_vectors, sorted_np2_lengths)[1]
            
            #######
            ###Loss
            #######
            p_rep = hidden_ec_p[sorted_back_to_original_list]
            del hidden_ec_p
            np1_rep = hidden_ec_np1[sorted_back_to_original_list_np1]
            del hidden_ec_np1
            np2_rep = hidden_ec_np2[sorted_back_to_original_list_np2]
            del hidden_ec_np2
            total_loss(hidden_ec, p_rep, np1_rep, np2_rep, lam_c, lam_w, margin).backward()

            if args.clip>0.0:                
                torch.nn.utils.clip_grad_norm(encoder.parameters(), args.clip)
                
            opt_e.step()
            
            step +=1
            batch_num +=1
            batch_size = original_batch_size
                        
        #if epoch%1 ==0:
        #    learning_rate= learning_rate * weight_decay
        
                
        elapsed = time.time()
        print("Elapsed time for epoch: ",elapsed-start)
        start = time.time()
    
        
        save_every_epoch(args, epoch)
    
    #writer.close()
    

if __name__ == '__main__':
    print("running")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.cuda is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)

    num_epochs = 7
    train(num_epochs, args)
    
    
