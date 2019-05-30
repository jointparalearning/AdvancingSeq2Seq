#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 23:42:23 2019

@author: TiffMin
"""
import argparse
import torch
from models.functions import *
import _pickle as cPickle
import pickle
import copy
from torch.nn.functional import cosine_similarity
from itertools import combinations
import numpy as np
import os
import random
from nltk.translate.bleu_score import sentence_bleu
from models.seq2seq_Luong import *


parser = argparse.ArgumentParser(description='Define trainig arguments')
parser.add_argument('-load_dir', '--loading_dir', type=str, metavar='', required=True, help="load model and start with it if there is a directory")
parser.add_argument('-e', '--epoch', type=int, metavar='', default = 10000, help="model type among 0,1,2")
parser.add_argument('-m', '--model_version', type=int, metavar='', required=True, help="model type among 0,1,2")
parser.add_argument('-sh', '--shuffle_scheme', type=int, metavar='', required=True, help="shuffle type among 0,1,2")
parser.add_argument('-spl', '--split_num', type=int, metavar='', required=True, help="split among 1,2,3,4,5")
parser.add_argument('-cos_only', '--cos_only', type=int, metavar='', required=False, help="split among 1,2,3,4,5")
parser.add_argument('-c', '--cuda', type=int, metavar='', required=False, help="split among 1,2,3,4,5")
parser.add_argument('-bin', '--binary', type = int, metavar='', required =True, help ="binary")
parser.add_argument('-con', '--conditional_copy', type = int, metavar='', required =True, help ="conditional copy")
parser.add_argument('-bi', '--bi', type = int,metavar='', required=True )


args = parser.parse_args()

lf_binary_entsRAW = cPickle.load(open("data/weihung_binary_lf.p", "rb"))

def load_model(args):
    load_dir = args.loading_dir
    model_type = args.model_version

    exec(open('data_prep/data_prepRAW_Shuffle.py').read(), globals(), globals())     
    
    #default hyperparameters 
    embed_dim = 128
    hidden_dim = 128
    latent_dim = 128
    AUTOhidden_dim = 128
    
    #initialize model
    #m_ver_dict = {0: 'models/plain.py', 1:'models/vae_ver1.py', 2:  'models/plain_explicit.py', 3:'models/vae_explicit.py'}
    #exec(open(m_ver_dict[model_type]).read(), globals(), globals())
    if model_type in [0,2]:
        encoder = EncoderRNN(vocab_size, hidden_dim)
        decoder = LuongAttnDecoderRNN('general',hidden_dim, vocab_size, bi=args.bi)
        AUTOdecoder = LuongAttnDecoderRNN('general',hidden_dim, vocab_size, bi=args.bi)
        
    elif model_type in [1,3]:
        raise NotImplementedError
        
    else:
        raise NotImplementedError
    
    #directories
    if args.epoch == 10000:
        encoder_dir = 'outputs/' + load_dir + '/encoder' + '.pt'  
        decoder_dir = 'outputs/' + load_dir + '/decoder' +  '.pt'
        auto_dir = 'outputs/' + load_dir + '/auto-decoder' '.pt'
    else:
        encoder_dir = 'outputs/' + load_dir + '/encoder_epoch' + str(args.epoch-1) + '.pt'  
        decoder_dir = 'outputs/' + load_dir + '/decoder_epoch' + str(args.epoch-1) + '.pt'
        auto_dir = 'outputs/' + load_dir + '/auto-decoder_epoch' + str(args.epoch-1) + '.pt'
    
    encoder.load_state_dict(torch.load(encoder_dir)) 
    decoder.load_state_dict(torch.load(decoder_dir)) 
    AUTOdecoder.load_state_dict(torch.load(auto_dir)) 
    
    #return encoder, decoder, AUTOdecoder, tokenized_eng_sentences, Qidx2LFIdxVec_dict
    return encoder, decoder, AUTOdecoder, tokenized_eng_sentences, Qidx2LFIdxVec_dict

def test(args):
 
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
        sentence_vectors,binary_vectors, x_lengths, labels, targ_lengths = [],[],[],[], []
        for idx in sorted_idxes:
            sent = sent2idxtensor(tokenized_eng_sentences[idx], idx)
            sentence_vectors.append(sent +[pad_token] * (max_length-len(sent)))
            if args.model_version in [2,3]:
                binary_vec = [SOS_token] + raw_question_binary_en[idx] + [EOS_token]
                binary_vectors.append(binary_vec + [pad_token] * (max_length-len(binary_vec)))
            else:
                pass
            x_lengths.append(len(sent))
            labels.append(Qidx2LFIdxVec_dict[idx] +[pad_token] * (max_y_length - len(Qidx2LFIdxVec_dict[idx])) )
            targ_lengths.append(len(labels[-1]))
        return torch.tensor(sentence_vectors, device=device),torch.tensor(binary_vectors, dtype=torch.float, device = device), torch.tensor(labels, device=device), x_lengths, sorted_idxes, targ_lengths

    global shuffle_scheme
    global split_num
    global binary
    global binary_vectors
    
    binary = args.binary
    if args.model_version in [2,3]:
        binary = 1
        assert not(args.conditional_copy is 1 and binary is 1)

    split_num, shuffle_scheme, save_dir = args.split_num, args.shuffle_scheme, args.loading_dir
    #encoder, decoder, AUTOdecoder, tokenized_eng_sentences, Qidx2LFIdxVec_dict = load_model(args)
    encoder, decoder, AUTOdecoder, tokenized_eng_sentences, Qidx2LFIdxVec_dict = load_model(args)
    print("models loaded!")
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        AUTOdecoder.cuda()
    encoder.eval(); decoder.eval(); AUTOdecoder.eval()
    
    #actual inference
    #current_loss_list = []
    test_batch_num = 0
    global batch_size
    batch_size = 32
    hidden_dim = 128
    
    hidden_ec_list = {}

    translation_pairs = {}; reconstruction_pairs={}
    sorted_idxes_dict = {}
    if args.cos_only is None:
        with torch.no_grad():
            while (test_batch_num) * batch_size < len(validation):
                if test_batch_num %10 ==0:
                    print("==================================================")
                    print("Test Batch Percent: ",100 *(test_batch_num) * batch_size/ len(validation), "%")
                sentence_vectors,binary_vectors, target, X_lengths, sorted_idxes, targ_lengths  = prepare_batch_test(test_batch_num)
                batch_size = sentence_vectors.shape[0]
                test_batch_num +=1
                translation_pairs[test_batch_num] = []; reconstruction_pairs[test_batch_num] = []
                input_out = sentence_vectors.view(batch_size,-1).data.cpu().numpy()
                
                # mask input to remove padding
                input_mask = np.array(input_out>0, dtype=int)
        
                # input and output in Variable form
                x = sentence_vectors
                y = target.view(batch_size, -1)
                
                # apply to encoder
                encoded, hidden_ec = encoder(x, X_lengths) # z is [batch_size, latent_size]
                #hidden_viewed = hidden_ec.view(batch_size, hidden_dim*2)
                hidden_viewed = torch.cat([encoded[:, -1, :hidden_dim], encoded[:, 0, hidden_dim:]  ], 1)
                assert len(sorted_idxes) == batch_size
                for i, idx in enumerate(sorted_idxes): 
                    hidden_ec_list[idx] = hidden_viewed[i]
                
                # get initial input of decoder
                SOS_token = 1
                decoder_input = torch.LongTensor([SOS_token] * batch_size)
                decoder_hidden = hidden_viewed  # Use last (forward) hidden state from encoder
                # We will flip because we have the batch first
                all_decoder_outputs = torch.zeros(batch_size, targ_lengths[0], decoder.output_size)
                # Shape: [5, 20, 298]
    
                if USE_CUDA:
                    all_decoder_outputs = all_decoder_outputs.cuda()
                    decoder_input = decoder_input.cuda()
                    
                out_list = []
                rec_out_list = []
                # Run through decoder one time step at a time
                for t in range(targ_lengths[0]):
                    decoder_output, decoder_hidden, decoder_attn = decoder(
                        decoder_input, decoder_hidden, encoded
                    )
                    out_list.append(decoder_output[:].max(1)[1].squeeze().cpu().data.numpy())
                    all_decoder_outputs[:, t] = decoder_output  # Store this step's outputs [5, 298]
                    decoder_input = target[:, t]  # Next input is current target
                
                ###Auto decoder 
                AUTOdecoder_hidden, AUTOdecoder_input, AUTOtarget_tensor, di = hidden_viewed, torch.tensor([SOS_token]*batch_size, device=device), sentence_vectors, 1
                use_teacher_forcing = False 
                
                if use_teacher_forcing:
                    for di in range(AUTOtarget_tensor.shape[1]):
                        if args.model_version==0:
                            AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                            AUTOdecoder_input, AUTOdecoder_hidden, encoded)
                        else:
                            AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                            AUTOdecoder_input, AUTOdecoder_hidden, di,  batch_size)
                        AUTOdecoder_input = AUTOtarget_tensor[:,di]  # Teacher forcing
                        di+=1
                        
                        
                        topv, topi = AUTOdecoder_output.topk(1) 
                        
                        rec_out_list.append(AUTOdecoder_output.topk(1)[1].cpu().data.numpy())
                        if di ==1:
                            tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])
                        else:
                            tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,  AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])], dim=1)
        
                else:
                    for di in range(AUTOtarget_tensor.shape[1]):
                        if args.model_version==0:
                            AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                            AUTOdecoder_input, AUTOdecoder_hidden, encoded)
                        else:
                            AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                            AUTOdecoder_input, AUTOdecoder_hidden, di,  batch_size)
                        topv, topi = AUTOdecoder_output.topk(1) 
                        AUTOdecoder_input = topi.squeeze().detach() 
                        rec_out_list.append(AUTOdecoder_input.cpu().data.numpy())
                        di +=1
                        if di ==1:
                            tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])
                        else:
                            tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,  AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])], dim=1)
            
                
                review_sent = AUTOtarget_tensor.view(-1)
                review_auto_out = tensor_of_all_AUTOdecoded_outputs.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2])
                
                
                # all_decoder_outputs.topk(1)[0].size() - [5,48,1]
                # For reference the topk 1, refers to 1 from the top. It always does it over the last dimension
                # (vocab_size) and keep the other dimensions as it is

                target = y.view(-1)               
           
                predicted = np.transpose(np.array(out_list))  
                predicted_auto = np.transpose(np.array(rec_out_list))
                translation_pairs[test_batch_num].append((predicted, y.cpu().numpy()))
                reconstruction_pairs[test_batch_num].append((predicted_auto, sentence_vectors.cpu().numpy()))
                sorted_idxes_dict[test_batch_num] =sorted_idxes
                batch_size = 32
        
        save_dir = "outputs/" + save_dir + "/validation_results"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        print("Now saving test results...")
        test_result = {'translation_pairs':translation_pairs, 'reconstruction_pairs': reconstruction_pairs}        
        cPickle.dump(test_result, open(save_dir+"/validation_result.p","wb"))
        cPickle.dump(hidden_ec_list, open(save_dir+"/hidden_ec_list.p","wb"))
        
        translation_pairs = clean_pairs(translation_pairs)
        acc_list = acc(translation_pairs)
        print(args.loading_dir, "exact match is :", np.mean(acc_list))
        
        syntax_acc_list = syntax_bleu_acc(translation_pairs, sorted_idxes_dict)
        print(args.loading_dir, "syntax acc is : ",  np.mean(syntax_acc_list[0]))
        print(args.loading_dir, "bleu mean is : ",  np.mean(syntax_acc_list[1]))
        cPickle.dump(syntax_acc_list[1], open(save_dir+"bleu_list.p", "wb"))
    
    

    if args.cos_only is 1:
        save_dir = "outputs/" + save_dir + "/validation_results"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        hidden_ec_list = cPickle.load(open(save_dir+"/hidden_ec_list.p","rb"))

    
        print("Now calculating cosine similarities...")
        hidden_ec_templ_q, hidden_ec_lf,hidden_ec_lf_templq, cos_sim_inside_same_templq,cos_sim_diff_tq_same_lf, \
    cos_sim_diff_tq_same_lf_keyistqpair, cos_sim_inside_same_lf, cos_sim_for_lf_pairs= cosine(hidden_ec_list)
        print("Now saving everything...")
                
        cPickle.dump(hidden_ec_templ_q, open(save_dir+"/hidden_ec_templ_q.p","wb"))
        cPickle.dump(hidden_ec_lf, open(save_dir+"/hidden_ec_lf.p","wb"))
        cPickle.dump(hidden_ec_lf_templq, open(save_dir+"/hidden_ec_lf_templq.p","wb"))
        
        cPickle.dump(cos_sim_inside_same_templq, open(save_dir+"/cos_sim_inside_same_templq.p","wb"))
        cPickle.dump(cos_sim_diff_tq_same_lf, open(save_dir+"/cos_sim_diff_tq_same_lf.p","wb"))
        cPickle.dump(cos_sim_diff_tq_same_lf_keyistqpair, open(save_dir+"/cos_sim_diff_tq_same_lf_keyistqpair.p","wb"))
    
        cPickle.dump(cos_sim_inside_same_lf, open(save_dir+"/cos_sim_inside_same_lf.p","wb"))
        cPickle.dump(cos_sim_for_lf_pairs, open(save_dir+"/cos_sim_for_lf_pairs.p","wb"))



def cosine(hidden_ec_list):
    #1. reprocess hidden_ec_list into dictionary 
    #already hidden_ec_list[q_idx] = hidden_ec of that q <- shape is just [256]
    #now make 1.1 hidden_ec_templ_q[templ_q_idx] = [hidden_ecs of that templ_q]
    #         1.2 hidden_ec_lf[lf_idx] = [hidden_ecs of that lf]
    hidden_ec_templ_q = {}
    for qidx in hidden_ec_list:
        if not(rev_unique_templ_q_dict[qidx] in hidden_ec_templ_q):
            hidden_ec_templ_q[rev_unique_templ_q_dict[qidx]] =  [hidden_ec_list[qidx]]
        else:
            hidden_ec_templ_q[rev_unique_templ_q_dict[qidx]].append(hidden_ec_list[qidx])
    
    print("hidden_ec_templ_q finished")
    
    hidden_ec_lf = {}
    for qidx in hidden_ec_list:
        if not(rev_unique_lf_dict[qidx] in hidden_ec_lf):
            hidden_ec_lf[rev_unique_lf_dict[qidx]] =  [hidden_ec_list[qidx]]
        else:
            hidden_ec_lf[rev_unique_lf_dict[qidx]].append(hidden_ec_list[qidx])

    print("hidden_ec_lf finished")

    hidden_ec_lf_templq = {}
    for tq in  hidden_ec_templ_q:
        lf = rev_lf2temp_q_dict[tq]
        if not(lf in hidden_ec_lf_templq):
            hidden_ec_lf_templq[lf] = {tq: hidden_ec_templ_q[tq]}
        else:
            hidden_ec_lf_templq[lf][tq] = hidden_ec_templ_q[tq]
    
    print("hidden_ec_lf_templq finished")
    
    assert hidden_ec_lf_templq.keys() == hidden_ec_lf.keys()
    #2. calculate the cosine similarities
    #calculate the cosine similarity between all choose 2's in each group of templ_q, lf
    #do average of the sum of the above
    
    #1-1. Calculate the average cosine distance inside each teml_q 
    cos_sim_inside_same_templq = {} #key is templ_q
    for i, tq in enumerate(hidden_ec_templ_q):
        cos_sim_inside_same_templq[tq] = 0.0
        pairs_num = 0
        cos_sim = 0
        leng = len(hidden_ec_templ_q[tq])
        if leng >=2:
            random.seed(i)
            random.shuffle(hidden_ec_templ_q[tq])
            sample_leng = 2*int(leng/2)
            for j in range(int(sample_leng/2)):
                pair =  (hidden_ec_templ_q[tq][2*j], hidden_ec_templ_q[tq][2*j+1])
                cos_sim += cosine_similarity(pair[0],pair[1],0).item()
                pairs_num += 1
            cos_sim = cos_sim/pairs_num
        else:
            cos_sim = None
        cos_sim_inside_same_templq[tq] = cos_sim
    same_tq_cos_avg = np.mean([v for lf, v in cos_sim_inside_same_templq.items() if not(v is None)])
    
    print("cos_sim_inside_same_templq finished")

    
    #1-2. Calculate the average cosine distance of each teml_q group within the same lf 
    cos_sim_diff_tq_same_lf ={} #key is lf, value is number 
    cos_sim_diff_tq_same_lf_keyistqpair ={} #key is lf, then ky is tq pair, then value is number
    for i, lf in enumerate(hidden_ec_lf_templq):
        cos_sim_diff_tq_same_lf[lf] = 0.0
        cos_sim_diff_tq_same_lf_keyistqpair[lf] = {}
        q_pairs_num = 0
        cos_sim = 0
        if len(hidden_ec_lf_templq[lf]) >= 2:
            for tq_pair in combinations(hidden_ec_lf_templq[lf],2):
                tq_1 , tq_2= tq_pair[0], tq_pair[1]
                tq_pairs_num =0.0
                tq_cos_sim = 0.0
                random.seed(i)
                random.shuffle(hidden_ec_lf_templq[lf][tq_1])
                random.seed(i)
                random.shuffle(hidden_ec_lf_templq[lf][tq_2])
                for q1 in hidden_ec_lf_templq[lf][tq_1][:10]:
                    for q2 in hidden_ec_lf_templq[lf][tq_2][:10] :
                        cos_sim += cosine_similarity(q1,q2,0).item()
                        tq_cos_sim += cosine_similarity(q1,q2,0).item()
                        q_pairs_num +=1
                        tq_pairs_num +=1
                cos_sim_diff_tq_same_lf_keyistqpair[lf][tq_pair] = tq_cos_sim/ tq_pairs_num
                
            cos_sim_diff_tq_same_lf[lf] = cos_sim/q_pairs_num
        else:
            cos_sim_diff_tq_same_lf[lf],cos_sim_diff_tq_same_lf_keyistqpair[lf]  = None , None
    
    diff_tq_same_lf_cos_avg = np.mean([v for lf, v in cos_sim_diff_tq_same_lf.items() if not(v is None)])
    print("cos_sim_diff_tq_same_lf finished")

    
    #1-3. Calculate the average cosine distance of each teml_q group within different lf 's
    #Pass for now
    
    #2-1. Calculate the average cosine distance inside each lf 
    cos_sim_inside_same_lf = {} #key is lf
    for lf in hidden_ec_lf:
        cos_sim_inside_same_lf[lf] = 0.0
        pairs_num = 0
        cos_sim = 0
        leng = len(hidden_ec_lf[lf])
        if len(hidden_ec_lf[lf]) >=2:
            random.seed(i)
            random.shuffle(hidden_ec_lf[lf])
            sample_leng = 2*int(leng/2)
            for j in range(int(sample_leng/2)):
                pair =  (hidden_ec_lf[lf][2*j], hidden_ec_lf[lf][2*j+1])
                cos_sim += cosine_similarity(pair[0],pair[1],0).item()
                pairs_num += 1
            cos_sim = cos_sim/pairs_num
        else:
            cos_sim = None
        cos_sim_inside_same_lf[lf] = cos_sim
    same_lf_cos_avg = np.mean([v for lf, v in cos_sim_inside_same_lf.items() if not(v is None)])
    
    #2-2. Calculate the average cosine distance among different lf pairs
    cos_sim_for_lf_pairs = {} #key is lf pair
    c = 0
    for lf_pair in combinations(hidden_ec_lf.keys(),2):
        cos_sim_for_lf_pairs[lf_pair] = 0.0
        q_pairs_num = 0
        cos_sim = 0.0
        lf_1, lf_2 = lf_pair[0], lf_pair[1]
        random.seed(c)
        try:
            lf_1_ec_list = random.sample(hidden_ec_lf[lf_1], 50)
        except:
            lf_1_ec_list =  hidden_ec_lf[lf_1]
        random.seed(c)
        try:
            lf_2_ec_list = random.sample(hidden_ec_lf[lf_2], 50)
        except:
            lf_2_ec_list = hidden_ec_lf[lf_2]
        for ec_1 in lf_1_ec_list:
            for ec_2 in lf_2_ec_list:
                cos_sim += cosine_similarity(ec_1,ec_2,0).item()
                q_pairs_num +=1
        cos_sim = cos_sim/q_pairs_num
        cos_sim_for_lf_pairs[lf_pair] = cos_sim
        c +=1

    distinct_lf_cos_avg = np.mean([v for lf_pair, v in cos_sim_for_lf_pairs.items()])
    
    print(args.loading_dir, 'same_tq_cos_avg: ', same_tq_cos_avg)
    print(args.loading_dir, 'diff_tq_same_lf_cos_avg : ', diff_tq_same_lf_cos_avg)
    
    return hidden_ec_templ_q, hidden_ec_lf,hidden_ec_lf_templq, cos_sim_inside_same_templq,cos_sim_diff_tq_same_lf, \
cos_sim_diff_tq_same_lf_keyistqpair, cos_sim_inside_same_lf, cos_sim_for_lf_pairs


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


if __name__ == '__main__':
    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.cuda is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)
    test(args)
    print("complete!")
