#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:05:13 2019

@author: TiffMin
"""
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from util_functions import numpy_to_var, toData, to_np, to_var, decoder_initial, update_logger
import time
import sys
import math
from numpy import inf
import nltk
import random
import os

from multiprocessing import cpu_count
import copy

import pickle as cPickle
from models.seq2seq_Luong import *

from CzENG_dataset import CzENG, MyCollator

from collections import OrderedDict, defaultdict
import argparse
from models.masked_cross_enropy import *


parser = argparse.ArgumentParser(description='Define trainig arguments')
parser.add_argument('-a', '--alpha', type = float, metavar='', required=True, help = "alpha")
parser.add_argument('-cos_only', '--cos_only', type = int, metavar='', default=0, help = "alpha")
parser.add_argument('-cos_alph', '--cos_alph', type = float, metavar='', default=0, help = "alpha")

parser.add_argument('-save_dir', '--save_dir', type = str, metavar='', required=True, help = "alpha")
parser.add_argument('-load_dir', '--load_dir', type = str, metavar='', default = '', help = "alpha")

parser.add_argument('-seed', '--seed', type = int, metavar='', default=0, help = "alpha")
parser.add_argument('-which_attn', '--which_attn', type = str, metavar='', default = 'general', help ="conditional copy")
parser.add_argument('-num_epochs', '--num_epochs', type = int, metavar='', default = 50, help = "cuda")
parser.add_argument('-hid_size', '--hid_size', type = int, metavar='', default = 256, help = "cuda")
parser.add_argument('-small_train', '--small_train', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-batch_size', '--batch_size', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-eng_only', '--eng_only', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-embed_size', '--embed_size', type = int, metavar='', default = 300, help ="conditional copy")
parser.add_argument('-cross_ent', '--cross_ent', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-prob_already', '--prob_already', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-weight_decay', '--weight_decay', type = float, metavar='', default = 1, help ="conditional copy")
parser.add_argument('-half_factor', '--half_factor', type = float, metavar='', default = 1, help ="conditional copy")
parser.add_argument('-train_size', '--train_size', type = int, metavar='', default = 2, help ="conditional copy") #1 of [1,2,5]
parser.add_argument('-test_yourself', '--test_yourself', type = int, metavar='', default = 0, help ="conditional copy") #1 of [1,2,5]
parser.add_argument('-fast_text', '--fast_text', type = int, metavar='', default = 0, help ="conditional copy") #1 of [1,2,5]
parser.add_argument('-word_vec', '--word2vec', type = int, metavar='', default = 0, help ="conditional copy") #1 of [1,2,5]
parser.add_argument('-teacher_forcing', '--teacher_forcing', type = float, metavar='', default = 1.0, help ="conditional copy") #1 of [1,2,5]


#Take last hidden state or mean 
parser.add_argument('-mean_or_last', '--mean_or_last', type = str, metavar='', default = 'last', help = "alpha")

parser.add_argument('-clip', '--clip', type = float, metavar='', default = 0, help = "alpha")

parser.add_argument('-dropout', '--dropout', type = str, metavar='', default = 'CzEng', help = "alpha")
parser.add_argument('-pad', '--pad', type = float, default = 0,metavar='')
parser.add_argument('-lr', '--lr', type = float, default = 0,metavar='')

parser.add_argument('-half_end', '--half_end', type=int, metavar='', default=36, help = "hyperparameters")
parser.add_argument('-half_start', '--half_start', type=int, metavar='', default=15, help = "hyperparameters")
parser.add_argument('-k', '--k', type = int, metavar='', required=True )

parser.add_argument('-v', '--result_verbose', type = int, metavar='', default = 0, help = "cuda")
parser.add_argument('-s_e_e', '--save_every_epoch', type = int, metavar='', default=0, help = "alpha")
parser.add_argument('-s_from', '--save_from', type = int, metavar='', default=100, help = "alpha")
parser.add_argument('-c', '--cuda', type = int, metavar='', default = 0, help = "cuda")


args = parser.parse_args()

def to_cuda(tensor):
    # turns to cuda
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

def joint_loss_tensors_f(alpha, translated_predicted, translated_actual, reconstructed_predicted,
                             reconstructed_actual):
    pad_token = 0
    temp_loss_function = nn.NLLLoss(ignore_index=pad_token)
    translation_loss = temp_loss_function(translated_predicted, translated_actual)
    reconstruction_loss = temp_loss_function(reconstructed_predicted, reconstructed_actual)
    total_loss = translation_loss + alpha * reconstruction_loss
    return translation_loss, reconstruction_loss, total_loss

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


def train(args):
    torch.manual_seed(args.seed)
    assert args.train_size in [1,2,5]
    assert args.word2vec + args.fast_text in [0,1]
    
    #load train
    datasets = OrderedDict()
    fast_text_en, fast_text_cz = None, None
    print("Reading data")
    
    train_file = 'data/populated_selected_training_0.1M.p'
    test_file = 'data/para_test_dict_0.1M.p'
    en_vocab_file = 'data/en_voc_subtitles_0.1m.p'
    cz_vocab_file = 'data/cz_voc_subtitles_0.1m.p'

    if args.word2vec == 1:
        fast_text_en = cPickle.load(open('data/en_word2vec_0.1m.p', 'rb'))


    
    datasets['train'] = CzENG(
            data_file = train_file,
            split='train')

    #load test 
    datasets['test'] = CzENG(
        data_file=test_file,
        split='test')

     
        #load vocab
    en_vocab = cPickle.load(open(en_vocab_file,'rb')); cz_vocab = cPickle.load(open(cz_vocab_file,'rb'))
    en_w2i, en_i2w = en_vocab["en_w2i"], en_vocab["en_rm_i2w"]
    cz_w2i, cz_i2w = cz_vocab["cz_w2i"], cz_vocab["cz_rm_i2w"]
    
    pad_token = en_w2i['PAD_token']; SOS_token = en_w2i['SOS_token'] ; EOS_token = en_w2i['EOS_token']
    en_vocab_size = len(en_w2i); cz_vocab_size = len(cz_w2i)

    print("Read all data!")

    #define hidden dim, etc 
    embed_size = args.embed_size
    #learning_rate = 0.0008
    learning_rate = args.lr
    AUTOhidden_dim = args.hid_size
    weight_decay = 0.99
    hidden_size = args.hid_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    save_directory = 'outputs/' + args.save_dir
    load_dir = 'outputs/' + args.load_dir
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    
    #encoder, decoder 
    encoder = EncoderRNN(en_vocab_size, hidden_size, embed_size, fast_text = args.fast_text + args.word2vec, fast_text_vec =fast_text_en)
    decoder = LuongAttnDecoderRNN(args.which_attn,hidden_size, cz_vocab_size, cz_vocab_size, embed_size,  bi=1, fast_text = args.fast_text, fast_text_vec =fast_text_cz)
    AUTOdecoder =LuongAttnDecoderRNN(args.which_attn, hidden_size, en_vocab_size, en_vocab_size,embed_size, bi =1, fast_text = args.fast_text, fast_text_vec =fast_text_cz)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        AUTOdecoder.cuda()
    
    epochs = range(num_epochs)
    if args.load_dir != '':
        trained_until = 29
        
        encoder_dir =  load_dir + '/encoder_ckpt_epoch' + str(trained_until) +'.pytorch'  
        decoder_dir = load_dir + '/decoder_ckpt_epoch' + str(trained_until) +'.pytorch'
        auto_dir = load_dir + '/auto_decoder_ckpt_epoch' + str(trained_until) +'.pytorch'
        
        encoder.load_state_dict(torch.load(encoder_dir)) 
        decoder.load_state_dict(torch.load(decoder_dir)) 
        AUTOdecoder.load_state_dict(torch.load(auto_dir)) 
        
        encoder.train(); decoder.train(); AUTOdecoder.train()
        epochs = range(trained_until+1, num_epochs)

    train_collate = MyCollator()
    test_collate = MyCollator(True)
    
    loss_function = joint_loss_tensors_f
    base_loss_function = nn.NLLLoss(ignore_index=pad_token)
    cos_loss_function = torch.nn.CosineEmbeddingLoss()
    
    reconstruction_pairs = {}
    translation_pairs = {}
    
    #current_loss_list = []
    start = time.time()
    i_sum = 0
    for epoch in epochs:
        avg_translation_loss = 0.0; avg_reconstruction_loss = 0.0; avg_cos_loss = 0.0; avg_multi_loss = 0.0
        translation_pairs[int(epoch + 1)] = []

        print("==================================================")
        print("Epoch ", epoch + 1)
        
        print("elapsed time for epoch: ", time.time()- start)
        start = time.time()

        
        opt_e = optim.Adam(params=encoder.parameters(), lr=learning_rate)
        opt_d = optim.Adam(params=decoder.parameters(), lr=learning_rate)
        opt_a = optim.Adam(params=AUTOdecoder.parameters(), lr=learning_rate)
        
        data_loader_train = DataLoader(
            dataset=datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_collate,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
            
        )
        
        timer = time.time()
        for i, batch_list in enumerate(data_loader_train):
            i_sum +=1
            if i %100000 ==1:
                print(str(i/62500 *100) ,"% of one epoch done")
                print("time per batch :", time.time()-timer)
            timer = time.time()

            
            #print('eng idxes: ', batch_list[0]['eng_idxes'])
            
            batch = {}
            batch['eng_idxes'] = torch.tensor([batch_list[i]['eng_idxes'] for i in range(len(batch_list))], dtype = torch.long)
            batch['cz_idxes'] = torch.tensor([batch_list[i]['cz_idxes'] for i in range(len(batch_list))], dtype = torch.long)
            batch['eng_leng'] = torch.tensor([batch_list[i]['eng_leng'] for i in range(len(batch_list))], dtype = torch.long)
            batch['cz_leng'] = torch.tensor([batch_list[i]['cz_leng'] for i in range(len(batch_list))], dtype = torch.long)
            batch['para_leng'] = torch.tensor([batch_list[i]['para_leng'] for i in range(len(batch_list))], dtype = torch.long)
            batch['para_idxes'] = torch.tensor([batch_list[i]['para_idxes'] for i in range(len(batch_list))], dtype = torch.long)

            
            sorted_lengths, sorted_idx = torch.sort(batch['eng_leng'], descending=True)
            batch['eng_idxes'] = batch['eng_idxes'][sorted_idx]
            batch['cz_idxes'] = batch['cz_idxes'][sorted_idx]
            batch['eng_leng'] = batch['eng_leng'][sorted_idx]
            batch['cz_leng'] = batch['cz_leng'][sorted_idx]
            cur_batch_size = len(batch['eng_leng'])
            #print("eng :", batch['eng_idxes'] )
            #print("cz :",batch['cz_idxes'] )
            
            input_vectors = to_cuda(batch['eng_idxes'])
            target_vectors = to_cuda(batch['cz_idxes'])

            #print("eng leng: ",batch['eng_leng'])
            #print("eng idxes: ",batch['eng_idxes'])


            opt_e.zero_grad()
            opt_d.zero_grad()
            opt_a.zero_grad()
            
            para_sorted_lengths, para_sorted_idx = torch.sort(batch['para_leng'], descending=True)
            para_vectors = batch['para_idxes'][sorted_idx].cuda()
              
            encoded, hidden_ec = encoder(input_vectors, sorted_lengths)
            #print("input num: ", input_vectors.shape[1])
            #print("encoded shape:", encoded.shape)
            #print("hidden_ec shape:", hidden_ec.shape)
            if math.isnan(encoded[0][0].data[0]):
                print("encoder broken!")
                sys.exit()
                break
            

            if args.mean_or_last == 'last':
                s1 = torch.cat([encoded[:, -1, :hidden_size], encoded[:, 0, hidden_size:]  ], 1)
                s1 = s1.contiguous()
            elif args.mean_or_last == 'mean':
                s1 = torch.mean(encoded, 1)
                s1 = s1.contiguous()
             
                
            #Main Decoder 
            decoder_in, s, w = decoder_initial(cur_batch_size);SOS_token = 1; decoder_in = torch.LongTensor([SOS_token] * cur_batch_size).cuda()
            decoder_hidden = s1
            out_list = []
            #print("target vectors size", target_vectors.size(1))
            for j in range(target_vectors.size(1)):  # for all sequences
                """
                decoder_in (Variable): [b]
                encoded (Variable): [b x seq x hid]
                input_out (np.array): [b x seq]
                s (Variable): [b x hid]
                """
                # 1st state
                
                decoder_output, decoder_hidden, decoder_attn = decoder(
                        decoder_in, decoder_hidden, encoded
                )
                #print("decoder output shape:", decoder_output.shape)
                #print("decoder output unsqueeze:", decoder_output.view(decoder_output.shape[0], 1, decoder_output.shape[1]))
                #decoder_output = decoder_output.cpu()
                if j ==0:
                    out = decoder_output.unsqueeze(1)
                else:
                    out = torch.cat([out,decoder_output.unsqueeze(1)],dim=1)
                if random.random() < args.teacher_forcing:
                    decoder_in = target_vectors[:, j]  # Next input is current target
                else:
                    decoder_in = out[:, -1].max(1)[1].cuda() 

                out_list.append(out[:, -1].max(1)[1].cpu().data.numpy())
                
            
            #AUTO DECODER
            AUTOdecoder_hidden, AUTOdecoder_input, AUTOtarget_tensor, di = s1, torch.tensor([SOS_token]*cur_batch_size, device=device).cuda(), para_vectors, 1
            if args.alpha > 0 and args.cos_only ==0:
                for di in range(AUTOtarget_tensor.shape[1]):
                    AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(AUTOdecoder_input, AUTOdecoder_hidden,
                                                                         encoded)
                    # print("AUTOdecoder_output: " + str(AUTOdecoder_output.shape))
                    AUTOdecoder_input = AUTOtarget_tensor[:, di]  # Teacher forcing
                    di += 1
                    #AUTOdecoder_output = AUTOdecoder_output.cpu()
                    if di == 1:
                        tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0], 1,
                                                                                    AUTOdecoder_output.shape[1])
                    else:
                        tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,
                                                                       AUTOdecoder_output.view(
                                                                           AUTOdecoder_output.shape[0], 1,
                                                                          AUTOdecoder_output.shape[1])], dim=1)
            #print("up until the loss!")
            #Now Calculate Loss
            target = target_vectors.contiguous()
            target = target.view(-1)
            pad_out = torch.softmax(out.view(-1, out.shape[2]), dim=1)
            pad_out = pad_out #+ 0.0001
            if args.pad == 1:
                pad_out = pad_out + 0.0001
            pad_out = torch.log(pad_out)
            
            if args.cross_ent ==1:
                #1. baseline 
                if args.alpha == 0:
                    if args.pad == 0:
                        joint_loss = masked_cross_entropy(
                            (out).contiguous(),
                            target_vectors.contiguous(),
                            batch['cz_leng'],
                            args.prob_already
                        )
                    else:
                        joint_loss = masked_cross_entropy(
                            (out).contiguous()+0.0001,
                            target_vectors.contiguous()+0.0001,
                            batch['cz_leng']+0.0001,
                            args.prob_already
                        )
                    avg_translation_loss +=  joint_loss
                 #Does cosine  
                elif args.alpha > 0 and args.cos_alph >0 :
                    para_encoder_vectors = batch['para_idxes'][para_sorted_idx].cuda(); para_encoder_lengths = batch['para_leng'][para_sorted_idx]
                    encoded_p, hidden_ec_p = encoder(para_encoder_vectors, para_encoder_lengths)
                    if args.mean_or_last == 'last':
                        p_rep = torch.cat([encoded_p[:, -1, :hidden_size], encoded_p[:, 0, hidden_size:]  ], 1)
                    else:
                        raise NotImplementedError
                    para_sorted_2_original_dict = {v:k for k, v in enumerate(para_sorted_idx)}
                    para_sorted_2_original_list = [-1]*len(para_sorted_2_original_dict)
                    for v, k in para_sorted_2_original_dict.items():
                        para_sorted_2_original_list[v] = k
                    p_rep = p_rep[para_sorted_2_original_list]
                        
                    cos_loss = cos_loss_function(p_rep, s1, torch.tensor([1.0]*cur_batch_size).cuda())
                    #3. only cosine 
                    if args.cos_only == 1:
                        translation_loss =  masked_cross_entropy(
                            (out).contiguous()+0.0001,
                            target_vectors.contiguous()+0.0001,
                            batch['cz_leng']+0.0001,
                            args.prob_already
                        )
                        joint_loss = translation_loss + args.alpha * args.cos_alph * cos_loss
                        avg_translation_loss += translation_loss; avg_reconstruction_loss +=   cos_loss
                        avg_cos_loss += cos_loss
                 
                
            else:
                #1. baseline 
                if args.alpha == 0:
                    target = target.cuda()
                    translation_loss = base_loss_function(pad_out, target)
                    joint_loss = translation_loss
                    avg_translation_loss += translation_loss
                #2. only gen
                elif args.alpha > 0 and args.cos_alph == 0:
                    review_sent = AUTOtarget_tensor.contiguous().view(-1)
                    review_auto_out = torch.softmax(tensor_of_all_AUTOdecoded_outputs.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2]), dim=1)
                    if args.pad == 1:
                        review_auto_out = review_auto_out + 0.0001
                    review_auto_out = torch.log(review_auto_out)
                    
                    translation_loss, reconstruction_loss, joint_loss = loss_function(args.alpha, pad_out, target,
                                                                                          review_auto_out,review_sent)
                    avg_translation_loss += translation_loss; avg_reconstruction_loss += reconstruction_loss
                    avg_multi_loss += reconstruction_loss
    
                
                #Does cosine                                                                            
                elif args.alpha > 0 and args.cos_alph >0 :
                    para_encoder_vectors = batch['para_idxes'][para_sorted_idx].cuda(); para_encoder_lengths = batch['para_leng'][para_sorted_idx]
                    encoded_p, hidden_ec_p = encoder(para_encoder_vectors, para_encoder_lengths)
                    if args.mean_or_last == 'last':
                        p_rep = torch.cat([encoded_p[:, -1, :hidden_size], encoded_p[:, 0, hidden_size:]  ], 1)
                    else:
                        raise NotImplementedError
                    para_sorted_2_original_dict = {v:k for k, v in enumerate(para_sorted_idx)}
                    para_sorted_2_original_list = [-1]*len(para_sorted_2_original_dict)
                    for v, k in para_sorted_2_original_dict.items():
                        para_sorted_2_original_list[v] = k
                    p_rep = p_rep[para_sorted_2_original_list]
                        
                    cos_loss = cos_loss_function(p_rep, s1, torch.tensor([1.0]*cur_batch_size).cuda())
                    #3. only cosine 
                    if args.cos_only == 1:
                        translation_loss = base_loss_function(pad_out, target)
                        joint_loss = translation_loss + args.alpha * args.cos_alph * cos_loss
                        avg_translation_loss += translation_loss; avg_reconstruction_loss +=   cos_loss
                        avg_cos_loss += cos_loss
                        
                        
                    #4. cosine + gen 
                    elif args.cos_only == 0:
                        review_sent = AUTOtarget_tensor.contiguous().view(-1)
                        review_auto_out = torch.softmax(tensor_of_all_AUTOdecoded_outputs.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2]), dim=1)
                        if args.pad == 1:
                            review_auto_out = review_auto_out + 0.0001
                        review_auto_out = torch.log(review_auto_out)
                        
                        translation_loss, reconstruction_loss, joint_loss = loss_function(args.alpha, pad_out, target,
                                                                                          review_auto_out,review_sent)
                        
                        joint_loss += args.alpha * args.cos_alph * cos_loss
                        avg_translation_loss += translation_loss; avg_reconstruction_loss += reconstruction_loss + args.cos_alph * cos_loss
                        avg_cos_loss += cos_loss; avg_multi_loss += reconstruction_loss
                    
                    
            
            #Gradient Descent
            joint_loss.backward()
            
            if args.clip>0:
                clip = args.clip
                
                torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
                torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
                torch.nn.utils.clip_grad_norm(AUTOdecoder.parameters(), clip)
            
            opt_e.step()
            opt_d.step()
            if args.alpha > 0 and args.cos_only ==0:
                opt_a.step()
# =============================================================================
#             
# # =============================================================================
# #             if (epoch+ 1)%args.k == (args.half_start+1)%args.k  and epoch + 1>args.half_start and epoch + 1<=args.half_end:
# #                 learning_rate = learning_rate * args.half_factor # weight decay
# # =============================================================================
#         
#             if (i_sum + 1) % args.k == 0:
#                 learning_rate = learning_rate * args.half_factor # weight decay
# 
#             
# =============================================================================
            if args.result_verbose == 1 and i %10000 ==500:
                predicted = np.transpose(np.array(out_list))
                y = target_vectors.cpu().view(cur_batch_size, -1)
            
                translation_pairs[epoch + 1].append((predicted, y.view(cur_batch_size, -1).cpu().numpy()))
                
                for i in range(min(5, y.shape[0])):
                    print("target: " +str(' '.join([cz_i2w[y[i][j].item()] for j in range(y.shape[1]) if y[i][j].item() != 0])))
                    temp_list = []
                    for j in range(predicted.shape[1]):
                        temp_list.append(cz_i2w[predicted[i][j]])
                    #print("predicted: " + str([LFidx2vocab[out_list[i].item()] for i in range(len(out_list))]))
                    print("predicted: " + str(' '.join(temp_list)))
                    print(' ')
                    
                tr_pairs = {epoch+1: translation_pairs[epoch + 1]}
                tr_pairs = clean_pairs(tr_pairs)
                avg_BLEU = 0.0
                avg_count = 0
                for e, pairs_list in tr_pairs.items():
                    #print("pairs list :", pairs_list)
                    for k in range(len(pairs_list)):
                        #candidate = pairs_list[k][0][1:-1]; reference = pairs_list[k][1][1:-1]
                        ref_len = len(pairs_list[k][1])
                        candidate = pairs_list[k][0][1:ref_len-1]; reference = pairs_list[k][1][1:-1]
                        #print("candiate:" , candidate)
                        #print("refenrece:", reference)
                        avg_BLEU += nltk.translate.bleu_score.sentence_bleu([reference], candidate)
                        avg_count  +=1 
                avg_BLEU = avg_BLEU / avg_count
                
                print("BLEU score: ", avg_BLEU * 100)
            
            
            
            
        if (epoch + 1)%args.k == (args.half_start+1)%args.k  and epoch + 1>args.half_start and epoch + 1<=args.half_end:
            learning_rate = learning_rate * args.half_factor # weight decay
        
        if (epoch + 1)>15 and epoch+1 % 10 == 0:
            learning_rate = learning_rate * args.weight_decay # weight decay
                                   
        avg_translation_loss = avg_translation_loss/(i+1); avg_reconstruction_loss =  avg_reconstruction_loss/(i+1)
        try:
            avg_multi_loss = avg_multi_loss / (i+1); avg_cos_loss = avg_cos_loss / (i+1)
        except:
            pass
        ####End of Training Loop
        ####Now Validation Loop
        data_loader_test = DataLoader(
                dataset=datasets['test'],
                batch_size=batch_size,
                shuffle=False,
                collate_fn=test_collate,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
        with torch.no_grad():
            translation_pairs[int(epoch + 1)] = []
            reconstruction_pairs[int(epoch + 1)] = []
            current_loss_list = []
            
            for i, batch_list in enumerate(data_loader_test):
# =============================================================================
#                 if args.which_test == 'CzENG': 
#                     if i> 50:
#                         break
#                     
# =============================================================================
                
                batch = {}
                batch['eng_idxes'] = torch.tensor([batch_list[i]['eng_idxes'] for i in range(len(batch_list))], dtype = torch.long)
                batch['cz_idxes'] = torch.tensor([batch_list[i]['cz_idxes'] for i in range(len(batch_list))], dtype = torch.long)
                batch['eng_leng'] = torch.tensor([batch_list[i]['eng_leng'] for i in range(len(batch_list))], dtype = torch.long)
                batch['cz_leng'] = torch.tensor([batch_list[i]['cz_leng'] for i in range(len(batch_list))], dtype = torch.long)
                
                
                sorted_lengths, sorted_idx = torch.sort(batch['eng_leng'], descending=True)
                batch['eng_idxes'] = batch['eng_idxes'][sorted_idx]
                batch['cz_idxes'] = batch['cz_idxes'][sorted_idx]
                batch['eng_leng'] = batch['eng_leng'][sorted_idx]
                batch['cz_leng'] = batch['cz_leng'][sorted_idx]
                cur_batch_size = len(batch['eng_leng'])
                
                input_vectors = to_cuda(batch['eng_idxes'])
                target_vectors = to_cuda(batch['cz_idxes'])

                    
                encoded, hidden_ec = encoder(input_vectors, sorted_lengths)
                if math.isnan(encoded[0][0].data[0]):
                    print("encoder broken!")
                    sys.exit()
                    break
                
                if args.mean_or_last == 'last':
                    s1 = torch.cat([encoded[:, -1, :hidden_size], encoded[:, 0, hidden_size:]  ], 1)
                    s1 = s1.contiguous()
                elif args.mean_or_last == 'mean':
                    s1 = torch.mean(encoded, 1)
                    s1 = s1.contiguous()

                #Main Decoder 
                decoder_in, s, w = decoder_initial(cur_batch_size);SOS_token = 1; decoder_in = torch.LongTensor([SOS_token] * cur_batch_size).cuda()
                decoder_hidden = s1
                out_list = []
                for j in range(target_vectors.size(1)):  # for all sequences
                    """
                    decoder_in (Variable): [b]
                    encoded (Variable): [b x seq x hid]
                    input_out (np.array): [b x seq]
                    s (Variable): [b x hid]
                    """
                    # 1st state
                    
                    decoder_output, decoder_hidden, decoder_attn = decoder(
                            decoder_in, decoder_hidden, encoded
                    )
                    decoder_output = decoder_output.cpu()
                    if j ==0:
                        out = decoder_output.unsqueeze(1).cpu()
                    else:
                        out = torch.cat([out,decoder_output.unsqueeze(1)],dim=1)
                    decoder_in = out[:, -1].max(1)[1].cuda() 
                    #out_list.append(out[:, -1].max(1)[1].squeeze().cpu().data.numpy())
                    out_list.append(out[:, -1].max(1)[1].cpu().data.numpy())
                
                #AUTO DECODER
                            
                AUTOdecoder_hidden, AUTOdecoder_input, AUTOtarget_tensor, di = s1, torch.tensor([SOS_token]*cur_batch_size, device=device), input_vectors, 1
                if args.alpha > 0 and args.cos_only ==0:
                    for di in range(AUTOtarget_tensor.shape[1]):
                        AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(AUTOdecoder_input, AUTOdecoder_hidden,
                                                                             encoded)
                        topv, topi = AUTOdecoder_output.topk(1)
                        AUTOdecoder_input = topi.squeeze(1).detach()
                        AUTOdecoder_output = AUTOdecoder_output.cpu()
                        di += 1
                        if di == 1:
                            tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0],
                                                                                        1,
                                                                                        AUTOdecoder_output.shape[1])
                        else:
                            tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,
                                                                           AUTOdecoder_output.view(
                                                                               AUTOdecoder_output.shape[0], 1,
                                                                               AUTOdecoder_output.shape[1])], dim=1)
                predicted = np.transpose(np.array(out_list))
                y = target_vectors.cpu().view(cur_batch_size, -1)
            
                translation_pairs[epoch + 1].append((predicted, y.view(cur_batch_size, -1).cpu().numpy()))
                try:
                    reconstruction_pairs[epoch + 1].append((tensor_of_all_AUTOdecoded_outputs
                                                            .topk(1)[1].squeeze(2).cpu().numpy(), input_vectors.cpu().numpy()))
                except:
                    pass
               
        #print(" yshape ", y.shape)
        #Verbose Results
        if args.result_verbose == 1:

            #for i in range(y.shape[0]):
            for i in range(min(5, y.shape[0])):
                print("target: " +str(' '.join([cz_i2w[y[i][j].item()] for j in range(y.shape[1]) if y[i][j].item() != 0])))
                temp_list = []
                for j in range(predicted.shape[1]):
                    temp_list.append(cz_i2w[predicted[i][j]])
                #print("predicted: " + str([LFidx2vocab[out_list[i].item()] for i in range(len(out_list))]))
                print("predicted: " + str(' '.join(temp_list)))
                print(' ')
            try:
                for i in range(min(5, y.shape[0])):
                    print("target: " +str(' '.join([en_i2w[input_vectors[i][j].item()] for j in range(input_vectors.shape[1]) if input_vectors[i][j].item() != 0])))
                    temp_list = []
                    for j in range(tensor_of_all_AUTOdecoded_outputs.topk(1)[1].squeeze(2).cpu().numpy().shape[1]):
                        temp_list.append(en_i2w[tensor_of_all_AUTOdecoded_outputs.topk(1)[1].cpu().numpy().squeeze(2)[i][j]])
                    #print("predicted: " + str([LFidx2vocab[out_list[i].item()] for i in range(len(out_list))]))
                    print("predicted: " + str(' '.join(temp_list)))
                    print(' ')
            except:
                pass
                        
            
        
        
        #Now Calculate BLEU and show validation examples
        tr_pairs = {epoch+1: translation_pairs[epoch + 1]}
        tr_pairs = clean_pairs(tr_pairs)
        avg_BLEU = 0.0
        
        avg_counter = 0
        for e, pairs_list in tr_pairs.items():
            #print("pairs list :", pairs_list)
            for k in range(len(pairs_list)):
                #candidate = pairs_list[k][0][1:-1]; reference = pairs_list[k][1][1:-1]
                ref_len = len(pairs_list[k][1])
                candidate = pairs_list[k][0][1:ref_len-1]; reference = pairs_list[k][1][1:-1]
                #print("candiate:" , candidate)
                #print("refenrece:", reference)
                if len(reference) < 4:
                    weights = [1.0/len(reference)]*len(reference)
                    #print("weights : ", weights)
                    BLEU_score = nltk.translate.bleu_score.sentence_bleu([reference], candidate, weights = weights)
                else:
                    BLEU_score = nltk.translate.bleu_score.sentence_bleu([reference], candidate)
                avg_BLEU += BLEU_score
                #print("current BLEU: ", BLEU_score)
                avg_counter += 1
        avg_BLEU = avg_BLEU / avg_counter
        
        print("BLEU score: ", avg_BLEU * 100)
        print("Training translation loss: ", avg_translation_loss)
        print("Training reconstruction loss: ", avg_reconstruction_loss)
        try:
            print("Training cos loss: ", avg_cos_loss)
            print("Training multi loss: ", avg_multi_loss)
        except:
            pass

            
        if args.save_every_epoch == 1 and epoch > args.save_from:
            final_dict = {'translation_pairs': translation_pairs, 'reconstruction_pairs': reconstruction_pairs}
            pkl_filname = os.path.join(save_directory, "validation_pairs_epoch%d" % epoch)
# =============================================================================
#             with open(pkl_filname, "wb") as f:
#                 cPickle.dump(final_dict, f)
# =============================================================================
                
            torch.save(encoder.state_dict(),
                       os.path.join(save_directory, "encoder_ckpt_epoch%i.pytorch" % epoch))
            torch.save(decoder.state_dict(),
                       os.path.join(save_directory, "decoder_ckpt_epoch%i.pytorch" % epoch))
            torch.save(AUTOdecoder.state_dict(),
                       os.path.join(save_directory, "auto_decoder_ckpt_epoch%i.pytorch" % epoch))
        
        elif epoch % 10 ==9:
            torch.save(encoder.state_dict(),
                       os.path.join(save_directory, "encoder_ckpt_epoch%i.pytorch" % epoch))
            torch.save(decoder.state_dict(),
                       os.path.join(save_directory, "decoder_ckpt_epoch%i.pytorch" % epoch))
            torch.save(AUTOdecoder.state_dict(),
                       os.path.join(save_directory, "auto_decoder_ckpt_epoch%i.pytorch" % epoch))


                
 
        
                
        
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.cuda.set_device(args.cuda)
    train(args)
