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
from models.masked_cross_enropy import *
from models.plain_method2 import *
from models.seq2seq_Luong import *
#from bert_serving.client import BertClient
from paraphrase_embedding_model import *


#from tensorboardX import SummaryWriter




#from models.plain import 
#from models.vae_ver1 import 
#from models. import 

parser = argparse.ArgumentParser(description='Define trainig arguments')

parser.add_argument('-sh', '--shuffle_scheme', type = int, metavar='', required =True, help ="saving directory of output file")
parser.add_argument('-spl', '--split_num', type=int, metavar='', required=True, help= "split num")
parser.add_argument('-m', '--model_version', type = int, metavar='', required=True, help = "model version - vae")
parser.add_argument('-a', '--alpha', type = float, metavar='', required=True, help = "alpha")
parser.add_argument('-kl', '--loss_version', type = int, metavar='', required=True, help = "loss version")
parser.add_argument('-save_dir', '--saving_dir', type=str, metavar='', required=True, help="saving directory of output file; not required but can specify if you want to.")
parser.add_argument('-bin', '--binary', type = int, metavar='', required =True, help ="use teacher forcing for training")
parser.add_argument('-con', '--conditional_copy', type = int, metavar='', required =True, help ="conditional copy")
parser.add_argument('-bi', '--bi', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-s', '--seed', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-qp', '--q_to_p', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-ac', '--auto_copy', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-s_e_e', '--save_every_epoch', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-s_from', '--save_from', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-word_vec', '--word2vec', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-word_vec_medical', '--word2vec_medical', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-word_vec_nonmedical', '--word2vec_nonmedical', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-word_vec_corpus_inputoutput', '--word2vec_corpus_inputoutput', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-word_vec_corpus_inputonly', '--word2vec_corpus_inputonly', type = int, metavar='', default = 0, help ="conditional copy")


parser.add_argument('-bert_sent', '--bert_sent', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-bert_word', '--bert_word', type = int, metavar='', default = 0, help ="conditional copy")


parser.add_argument('-cos_obj', '--cos_obj', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-cos_hinge', '--cos_hinge', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-cos_alph', '--cos_alph', type = float, metavar='', default = 1.0, help ="conditional copy")
parser.add_argument('-cos_only', '--cos_only', type = int, metavar='', default = 0)
parser.add_argument('-margin', '--margin', type = float, metavar='', default = 1)


##Optional Arguments
parser.add_argument('-H', '--hyperparams', type=str, metavar='', required=False, help = "hyperparameters")
parser.add_argument('-load_dir', '--loading_dir', type=str, metavar='', required=False, help="load model and start with it if there is a directory")
parser.add_argument('-c', '--cuda', type = int, metavar='', required=False, help = "cuda")

parser.add_argument('-clip', '--clip', type = float, metavar='', default = 0.0, help = "cuda")

parser.add_argument('-pad', '--pad', type = int, metavar='', default = 1, help = "cuda")

parser.add_argument('-skip_val', '--skip_validation', type = int, metavar='', default = 0, help = "cuda")

parser.add_argument('-down_sample_perc', '--down_sample_perc', type = float, metavar='', default = 1.0, help = "cuda")

parser.add_argument('-proj_use', '--proj_use', type = int, metavar='', default = 0, help = "cuda")
parser.add_argument('-proj_basis_dim', '--proj_basis_dim', type = int, metavar='', default = 0, help = "cuda")

parser.add_argument('-gray', '--gray', type = int, metavar='', default = 0, help = "cuda")

parser.add_argument('-GRAN', '--GRAN', type = int, metavar='', default = 0, help = "cuda")
parser.add_argument('-load_GRAN', '--load_GRAN', type = str, metavar='', default = '', help = "cuda")
parser.add_argument('-freeze_Enc', '--freeze_Enc', type = int, metavar='', default =0, help = "cuda")

parser.add_argument('-paragen_pretrain', '--paragen_pretrain', type = int, metavar='', default =0, help = "cuda")
parser.add_argument('-load_paragen', '--load_paragen', type = str, metavar='', default ='', help = "cuda")


args = parser.parse_args()

#m_ver_dict = {0: 'models/plain.py', 1:'models/vae_ver1.py', 2:  'models/plain_explicit.py', 3:'models/vae_explicit.py', 4:'models/plain_method2.py'}
m_ver_dict = {0: 'models/plain_method2.py', 1:'models/plain_method2.py', 2:  'models/plain_explicit.py', 3:'models/vae_explicit.py'}


torch.manual_seed(1000*args.seed)

gray_file = ''
if args.gray == 1:
    gray_file = '/scratch/symin95/gray_emrqa_outputs/'


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

def train(num_epochs, args):
    #if args.q_to_p == 1:
    #    num_epochs = 20
    def acc_bowser(dict_pairs):
        try:
            tr_pairs = dict_pairs['translation_pairs']
        except:
            tr_pairs = dict_pairs['pairs_dict']['translation_pairs']
        tr_pairs = clean_pairs(tr_pairs)
        counter = 0
        for k, pairs_list in tr_pairs.items():
            assert k == epoch
            acc = sum([list(tup[0]) == list(tup[1]) for tup in pairs_list])/len(pairs_list)
            counter +=1
        assert counter==1
        return acc
    def log(args, epoch):
        file_name = gray_file + "outputs/" + args.saving_dir + "/logs.txt"
        file = open(file_name, "w")
        file.write("ran till epoch: " + str(epoch) + "\n")
        file.write("model: "+str(args.model_version)+ "\n" )
        file.write("alpha: "+ str(args.alpha)+ "\n")
        file.write("split num: " + str(args.split_num)+ "\n")
        file.write("shuffle scheme: " + str(args.shuffle_scheme)+ "\n")
        file.write("Hyperparameters are like the following: "+ "\n")
        file.write("      learning rate: "+ str(learning_rate)+ "\n")
        file.write("      weight decay: " + str(weight_decay)+ "\n")
        file.write("      batch size: " + str(batch_size)+ "\n")
        file.write("      teacher forcing ratio: " + str(teacher_forcing_prob)+ "\n")
        file.write("      embed dim: " + str(embed_dim)+ "\n")
        file.write("      latent dim: " + str(latent_dim)+ "\n")
        file.write("      hidden dim: " + str(hidden_dim)+ "\n")
        file.write("      AUTO hidden dim: " + str(AUTOhidden_dim)+ "\n")
        
        if not(kl_bool in [0,1]):
            raise NotImplementedError
        elif kl_bool == 1:
            file.write("loss: kl used in VAE autodecoder")
        else:
            file.write("loss: kl not used anywhere")
        file.close()

    def save(trans_pairs, rec_pairs, args):
        torch.save(encoder.state_dict(), gray_file + 'outputs/' + args.saving_dir + '/encoder.pt')
        torch.save(decoder.state_dict(),gray_file + 'outputs/' + args.saving_dir + '/decoder.pt')
        torch.save(AUTOdecoder.state_dict(),gray_file + 'outputs/' + args.saving_dir + '/auto-decoder.pt')
        pairs_dict = {'translation_pairs': trans_pairs, 'reconstruction_pairs': rec_pairs}
        loss_dict = {'training':training_loss_dict, 'validation': validation_loss_dict}
        final_dict = {'pairs_dict':pairs_dict, 'loss_dict':loss_dict}
        cPickle.dump(final_dict, open(gray_file + 'outputs/' + args.saving_dir + '/loss_list.p',"wb"))    
        
       
    def save_every_epoch(trans_pairs, rec_pairs, args, epoch):
        torch.save(encoder.state_dict(),gray_file + 'outputs/' + args.saving_dir + '/encoder_epoch' + str(epoch) +  '.pt')
        torch.save(decoder.state_dict(),gray_file + 'outputs/' + args.saving_dir + '/decoder_epoch' + str(epoch) +  '.pt')
        torch.save(AUTOdecoder.state_dict(),gray_file +'outputs/' + args.saving_dir + '/auto-decoder_epoch' + str(epoch) +  '.pt')
        pairs_dict = {'translation_pairs': trans_pairs, 'reconstruction_pairs': rec_pairs}
        loss_dict = {'training':training_loss_dict, 'validation': validation_loss_dict}
        final_dict = {'pairs_dict':pairs_dict, 'loss_dict':loss_dict}
        cPickle.dump(final_dict, open(gray_file + 'outputs/' + args.saving_dir + '/loss_list.p',"wb"))    
        
    
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
            if binary is 1:
                binary_vec = [SOS_token] + raw_question_binary_en[idx]  + [EOS_token]
                assert len(binary_vec) == len(sent), str(idx)
                binary_vectors.append(binary_vec + [pad_token] * (max_length-len(binary_vec)))
            else:
                pass
            x_lengths.append(len(sent))
            labels.append(Qidx2LFIdxVec_dict[idx] +[pad_token] * (max_y_length - len(Qidx2LFIdxVec_dict[idx])) )
        return torch.tensor(sentence_vectors, device=device), torch.tensor(binary_vectors, dtype=torch.float, device = device), torch.tensor(labels, device=device), x_lengths, sorted_idxes, rel_sorted_idxes
                
    def prepare_batch_test(test_batch_num):
        if (test_batch_num+1)*batch_size <= len(validation_sampled):
            end_num = (test_batch_num+1)*batch_size 
        else:
            end_num = len(validation_sampled)
            #batch_size = end_num - batch_num*batch_size
        sorted_idxes = sorted(validation_sampled[test_batch_num*batch_size:end_num], key = lambda idx: len(sent2idxtensor(tokenized_eng_sentences[idx], idx)), reverse=True)
        rel_sorted_idxes = sorted(range(test_batch_num*batch_size,end_num), key = lambda idx: len(sent2idxtensor(tokenized_eng_sentences[validation_sampled[idx]], idx)), reverse=True)

        #print("sorted idxes:" + str(sorted_idxes))
        max_length = len(sent2idxtensor(tokenized_eng_sentences[sorted_idxes[0]], sorted_idxes[0]))
        max_y_length = max([len(Qidx2LFIdxVec_dict[idx]) for idx in sorted_idxes])
        sentence_vectors,binary_vectors, x_lengths, labels = [],[],[],[]
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
        return torch.tensor(sentence_vectors, device=device),torch.tensor(binary_vectors, dtype=torch.float, device = device), torch.tensor(labels, device=device), x_lengths, rel_sorted_idxes
    

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
        return torch.tensor(p_vectors, device = device), [p_idx[0] for p_idx in p_idxes], torch.tensor(return_sorted_p_vectors, device = device),sorted_back_to_original_list, sorted_p_lengths
        
     
    def sample_negatives(sorted_idxes):
        neg_idxes = []
        neg_p_vectors = []
        
        SOS_token = 1; EOS_token = 2
        for idx in sorted_idxes:
            #if neg_q_to_p[idx] != []:
            counter = 0
            succeed = 0
            while succeed ==0 and counter <=100:
                temp_idx = random.sample(seventy_percent_idxes, 1)
                if not(temp_idx[0] in q_to_p[idx] or temp_idx[0] == idx):
                    succeed = 1
                    neg_p_idx = temp_idx
                    neg_idxes.append(neg_p_idx)
                counter +=1
                
            if succeed == 0:
                raise Exception('Did not succeed in sampling a negative example')
                
        max_neg_length = 2 + max([len(tokenized_eng_sentences[neg_idx[0]]) for neg_idx in neg_idxes])
        temp_neg_vectors = []
        for neg_idx in neg_idxes:
            neg_vector = [SOS_token] + [Qvocab2idx[token] for token in tokenized_eng_sentences[neg_idx[0]]]+[EOS_token]
            temp_neg_vectors.append(neg_vector)
            neg_vector = neg_vector + [pad_token] * (max_neg_length - len(neg_vector))
            neg_p_vectors.append(neg_vector)
            
        
        #sort and back
        sorted_neg_vectors = sorted(temp_neg_vectors, key=len, reverse=True)
        original_relative_idx_sorted = sorted(range(len(temp_neg_vectors)), key=lambda k: temp_neg_vectors[k], reverse=True)
        sorted_back_to_original_dict = {idx:i for i,idx in enumerate(original_relative_idx_sorted)}
        sorted_back_to_original_list = [-1] * len(neg_idxes)
        for idx, i in sorted_back_to_original_dict.items():
            sorted_back_to_original_list[idx] = i
        sorted_neg_lengths = [len(sorted_l) for sorted_l in sorted_neg_vectors]
        return_sorted_neg_vectors = [sorted_inp + [pad_token] * (max_neg_length - len(sorted_inp)) for sorted_inp in sorted_neg_vectors]
        
        return torch.tensor(return_sorted_neg_vectors, device = device),sorted_back_to_original_list, sorted_neg_lengths

        
        
    
    def joint_loss_nll(alpha, translated_predicted, translated_actual, reconstructed_predicted, reconstructed_actual, kl_true=False):
        #total_loss = 0.0
        temp_loss_function = nn.NLLLoss(ignore_index = pad_token)
        translation_loss = temp_loss_function(translated_predicted, translated_actual)
        reconstruction_loss = temp_loss_function(reconstructed_predicted, reconstructed_actual)
        if kl_true == False:
            total_loss = translation_loss + alpha* reconstruction_loss
            return translation_loss, reconstruction_loss, total_loss
        else:
            total_loss = translation_loss + alpha* (reconstruction_loss+kl_anneal(step)*kl)
            return translation_loss, reconstruction_loss, total_loss

    def kl_anneal(step_num):
        k = 0.0025; x0 = 2500
        return float(1/(1+np.exp(-k*(step_num-x0))))
  
    
    #writer = SummaryWriter()
    global shuffle_scheme
    global split_num
    global binary
    cuda, m_ver, alpha, shuffle_scheme, kl_bool, save_dir, load_dir, binary = args.cuda, args.model_version, args.alpha, args.shuffle_scheme, args.loss_version, args.saving_dir, args.loading_dir, args.binary
    
    if m_ver in [2,3]:
        binary = 1
        assert not(args.conditional_copy is 1 and binary is 1)

    
    #assert args.split_num in [1,2,3,4,5]
    split_num = args.split_num
        
    ###make folder for save_dir
    save_dir = gray_file + "outputs/" + save_dir 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    ###load the required arguments first
    #load shuff_scheme
    if shuffle_scheme in [0,1,2]:
        exec(open('data_prep/data_prepRAW_Shuffle.py').read(), globals(), globals())        
    else:
        raise ValueError('shuffling scheme argument wrong')
    
    #load alpha     
    alpha = alpha 
    temp_loss_function = nn.NLLLoss(ignore_index = pad_token)
    
    ###load optional arguments and model
    #load hyperparameters
    global batch_size 
    global embed_dim 
    global hidden_dim 
    global latent_dim 
    global hidden_dim 
    global AUTOhidden_dim 
    global teacher_forcing_prob
    global weight_decay 
    global learning_rate
    global kl
    global binary_vectors
    kl = None
        
    if args.hyperparams is not None:
        exec(open('hyperparams/'+args.hyperparams).read(), globals(), globals())
    else:
        #exec(compile(open(file).read(), file, 'exec'))
        exec(open('hyperparams/default_hyperparams.py').read(), globals(), globals()) 
    original_batch_size = batch_size
    
    word_vectors = None
    assert args.word2vec_medical + args.word2vec_nonmedical in [0,1]
    if args.word2vec == 1:
        if args.word2vec_medical == 1:
            #word_vectors = cPickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/Refactored_Tiffany/data/emrqa_word2vec.p','rb'))
            word_vectors = cPickle.load(open('data/emrqa_word2vec.p','rb'))
        elif args.word2vec_nonmedical == 1:
            #word_vectors = cPickle.load(open('/data/scratch-oc40/symin95/github_lf/logicalforms/Refactored_Tiffany/data/emrqa_nonmedical_word2vec.p','rb'))
            word_vectors = cPickle.load(open('data/emrqa_nonmedical_word2vec.p','rb'))
        elif args.word2vec_corpus_inputoutput ==1:
            word_vectors = cPickle.load(open('corpus_w2v/sh' + str(args.shuffle_scheme) + 'spl' + str(args.split_num) + 'dim300_inputoutput_w2v.p','rb'))
        elif args.word2vec_corpus_inputonly ==1:
            word_vectors = cPickle.load(open('corpus_w2v/sh' + str(args.shuffle_scheme) + 'spl' + str(args.split_num) + 'dim300_inputonly_w2v.p','rb'))


    #load model version
    assert m_ver in m_ver_dict
    exec(open(m_ver_dict[m_ver]).read(), globals(), globals())
    #initialize model and load if needed 
    if m_ver in [0,2,4]: #DID UNTIL HERE!
        if args.GRAN ==0:
            encoder = CopyEncoderRAW(hidden_dim, vocab_size, args.word2vec, word_vectors, args.proj_use, args.proj_basis_dim)
            encoder.train()
        else:
            encoder = ParaEncoderGRAN(hidden_dim, vocab_size)
            if args.load_GRAN != '':
                encoder.load_state_dict(torch.load(gray_file + 'outputs/' + args.load_GRAN + '/encoder_epoch6.pt')) 
        decoder = CopyDecoder(vocab_size, embed_dim*2, hidden_dim*2, bi =args.bi, bert_sent = args.bert_sent, word2vec = args.word2vec, word_vectors = word_vectors) #added word2vec and 
        #AUTOdecoder = AutoDecoder(AUTOhidden_dim, vocab_size)
        if args.auto_copy == 1:
            AUTOdecoder = CopyDecoder(vocab_size, embed_dim*2, hidden_dim*2, bi =args.bi)
        else:
            AUTOdecoder = LuongAttnDecoderRNN('general', hidden_dim, vocab_size, bi =args.bi, bert_sent = args.bert_sent)
    
    elif m_ver in [1,3]:
        encoder = CopyEncoderRAW(hidden_dim, vocab_size)
        decoder = CopyDecoder(vocab_size, embed_dim*2, hidden_dim*2, bi =args.bi)
        AUTOdecoder = VAEDecoder(AUTOhidden_dim, vocab_size, latent_dim, bi=args.bi)
        
    else:
        raise NotImplementedError
        
    if args.load_paragen !='':
        encoder.load_state_dict(torch.load(gray_file + 'outputs/' + args.load_paragen + '/encoder_epoch15.pt')) 
        AUTOdecoder.load_state_dict(torch.load(gray_file + 'outputs/' + args.load_paragen + '/auto-decoder_epoch15.pt'))
        encoder.train()
        AUTOdecoder.train()       

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        AUTOdecoder.cuda()

    
    #load save_dir
    
    #load loss function
    loss_function = joint_loss_nll  
    if args.cos_hinge == 1:
        temp_coss_loss = torch.nn.CosineEmbeddingLoss()
        def hinge_cos_loss(anchors, positives, negatives, margin):
            total_loss = 0.0
            ones = torch.tensor([1.0]).cuda()
            for i in range(anchors.shape[0]):
                anchor = anchors[i]; positive = positives[i]; negative = negatives[i]
                total_loss += max((temp_coss_loss(anchor.view(1,-1), positive.view(1,-1), ones)-1) - (temp_coss_loss(anchor.view(1,-1), negative.view(1,-1), ones)-1) + margin, torch.tensor(0.0).cuda())
            return total_loss/anchors.shape[0]
        
 
        #cos_loss_function = hinge_cos_loss
    else:
        cos_loss_function = torch.nn.CosineEmbeddingLoss()
        
    #load model 
    if args.bert_sent ==1 or args.bert_word ==1:
        total_dict = cPickle.load(open('bertlists/bert_split_total_dict.p', 'rb'))
        trainq, valq, testq = total_dict[split_num][0], total_dict[split_num][1], total_dict[split_num][2]
        #convert these to numpy array, so that rel_sorted_idxes are applicable 
        training_encoded = np.squeeze(np.array([v[0].asnumpy() for k, v in trainq.items()]), axis=1)
        training_hidden = np.squeeze(np.array([v[1].asnumpy() for k, v in trainq.items()]), axis=1)
        
        validation_encoded =np.squeeze(np.array([v[0].asnumpy() for k, v in valq.items()]), axis=1)
        validation_hidden = np.squeeze(np.array([v[1].asnumpy() for k, v in valq.items()]), axis=1)
            
    if load_dir:
        trained_until = 15
# =============================================================================
#         encoder_dir = 'outputs/' + load_dir + '/encoder.pt'  
#         decoder_dir = 'outputs/' + load_dir + '/decoder.pt'
#         auto_dir = 'outputs/' + load_dir + '/auto-decoder.pt'
# =============================================================================
        
        encoder_dir = gray_file + 'outputs/' + load_dir + '/encoder_epoch' + str(trained_until) +'.pt'  
        decoder_dir = gray_file + 'outputs/' + load_dir + '/decoder_epoch' + str(trained_until) +'.pt'
        auto_dir = gray_file + 'outputs/' + load_dir + '/auto-decoder_epoch' + str(trained_until) +'.pt'
        
        encoder.load_state_dict(torch.load(encoder_dir)) 
        decoder.load_state_dict(torch.load(decoder_dir)) 
        AUTOdecoder.load_state_dict(torch.load(auto_dir)) 
        
        encoder.train(); decoder.train(); AUTOdecoder.train()
        #loss_dict = cPickle.load(open('outputs/' + load_dir + "/loss_list.p", "rb"))
# =============================================================================
#         try:
#             trained_until = int(list(loss_dict['translation_pairs'].keys())[-1]/100)*100
#         except: 
#             trained_until = int(list(loss_dict['pairs_dict']['translation_pairs'].keys())[-1]/100)*100
#             
# =============================================================================
        
        epochs = range(trained_until+1, num_epochs)
        reconstruction_pairs = {}; translation_pairs = {}
# =============================================================================
#         try:
#             reconstruction_pairs = loss_dict['translation_pairs']
#             translation_pairs = loss_dict['reconstruction_pairs']
#         except:
#             reconstruction_pairs = loss_dict['pairs_dict']['translation_pairs']
#             translation_pairs = loss_dict['pairs_dict']['reconstruction_pairs']
#         
# =============================================================================
    else:
        epochs = range(num_epochs)
        reconstruction_pairs = {}
        translation_pairs = {}
    
    #actual training starts
    
    training_loss_dict = {'reconstruction': [], 'translation':[], 'joint':[], 'cos':[]}
    validation_loss_dict = {'reconstruction': [], 'translation':[], 'joint':[]}
    
    if args.q_to_p is 1:
        q_to_p = cPickle.load(open('data/q_to_p_dict.p','rb'))
# =============================================================================
#         if args.cos_hinge ==1:
#             neg_q_to_p = {q: [] for q in q_to_p}
#             for q in q_to_p:
#                 qs_paras = q_to_p[q]
#                 neg_q_to_p[q] = [neg_p for neg_p in seventy_percent_idx_dict if not(neg_p in qs_paras or neg_p == q)]
#                 
# =============================================================================
                
    
    if kl_bool ==1:
        training_loss_dict['kl'] = []
        validation_loss_dict['kl'] = []
    start = time.time()
    step = 0
    for epoch in epochs:
        if args.freeze_Enc ==0:
            opt_e = optim.Adam(params=encoder.parameters(),  lr=learning_rate)
        opt_d = optim.Adam(params=decoder.parameters(),  lr=learning_rate)
        opt_a = optim.Adam(params=AUTOdecoder.parameters(), lr=learning_rate)

                
        batch_num = 0         
        current_loss_list = []
        print("==================================================")
        print("Epoch ",epoch)
        avg_translation_loss, avg_reconstruction_loss, avg_joint_loss, avg_kl, avg_cos_loss, avg_multi_loss = torch.tensor(0.0).cuda(),  torch.tensor(0.0).cuda(),  torch.tensor(0.0).cuda(),  torch.tensor(0.0).cuda(),torch.tensor(0.0).cuda(),torch.tensor(0.0).cuda() 
        while (batch_num) * batch_size < len(training_sampled):
            if batch_num % 100 ==0:
                print("==================================================")
                print("Batch Num: ",batch_num)
                print("Batch Percent: ",100 *(batch_num) * batch_size/ len(training_sampled), "%")
            sentence_vectors, binary_vectors, target, X_lengths, sorted_idxes, rel_sorted_idxes = prepare_batch_training(batch_num)
            if args.q_to_p is 1:
                p_vectors, p_idxes, sorted_p_vectors,original_relative_idx_sorted, sorted_p_lengths = prepare_q_to_p_training(batch_num)
                
            #TODO: 이 밑으로 autoencoder같은거에 args.q_to_p하고 넣고 등등
            
            #batch_size = sentence_vectors.shape[0]
            batch_size = sentence_vectors.shape[0]
            if args.freeze_Enc ==0:
                opt_e.zero_grad()
            opt_d.zero_grad()
            opt_a.zero_grad()
    
            # obtain batch outputs
            input_out = sentence_vectors.view(batch_size,-1).data.cpu().numpy()
        
            # mask input to remove padding
            input_mask = np.array(input_out>0, dtype=int)
    
            # input and output in Variable form
            x = sentence_vectors
            y = target.view(batch_size, -1)
    
            # apply to encoder
            if args.bert_sent == 0:
                encoded, hidden_ec = encoder(x, X_lengths) # z is [batch_size, latent_size]
            else:
                encoded = torch.tensor(training_encoded[rel_sorted_idxes][:, :max(X_lengths), :]).cuda()
                hidden_ec = torch.tensor(training_hidden[rel_sorted_idxes]).cuda()
            
            # get initial input of decoder
            decoder_in, s, w = decoder_initial(batch_size)
            SOS_token = 1
            decoder_in = torch.LongTensor([SOS_token] * batch_size)
            ######BILSTM 
            if args.bert_sent == 0 and args.GRAN==0:
                if args.bi==0:
                    s1 = encoded[:, -1, :] 
                else:
                    s1 = torch.cat([encoded[:, -1, :hidden_dim], encoded[:, 0, hidden_dim:]  ], 1) # b x seq x 2*hidden 
            else:
                s1 = hidden_ec
            #decoder_in = decoder_in.cpu()
            if torch.cuda.is_available():
                decoder_in = decoder_in.cuda()
            
            
            # out_list to store outputs
            use_teacher_forcing = True if random.random() < teacher_forcing_prob else False
            out_list=[]
            for j in range(y.size(1)): # for all sequences
                """
                decoder_in (Variable): [b]
                encoded (Variable): [b x seq x hid]
                input_out (np.array): [b x seq]
                s (Variable): [b x hid]
                """
                #print("encoded object", encoded)
                #assert binary_vectors.size(1) == y.size(1)
                # 1st state
                if j==0:
                    out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                    encoded_idx=input_out, prev_state= s1,
                                    weighted=w, order=j, train_mode=True)
                   
                else:
                    tmp_out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                    encoded_idx=input_out, prev_state= s,
                                    weighted=w, order=j, train_mode=True)

                    out = torch.cat([out,tmp_out],dim=1)
                    
                if not(use_teacher_forcing):
                    decoder_in = out[:,-1].max(1)[1].squeeze() # train with sequence outputs
                else:
                    decoder_in = y[:,j] # train with ground truth
                out_list.append(out[:,-1].max(1)[1].squeeze().cpu().data.numpy())
                        
            ###AUTO DECODER
            if args.q_to_p == 0:
                AUTOdecoder_hidden, AUTOdecoder_input, AUTOtarget_tensor, di = s1, torch.tensor([SOS_token]*batch_size, device=device), sentence_vectors, 1
            else:
                AUTOdecoder_hidden, AUTOdecoder_input, AUTOtarget_tensor, di = s1, torch.tensor([SOS_token]*batch_size, device=device), p_vectors, 1

            use_teacher_forcing = True if random.random() < teacher_forcing_prob else False
            
            if args.auto_copy == 0:
                if use_teacher_forcing:
                    for di in range(AUTOtarget_tensor.shape[1]):
                        if m_ver == 0:
                            AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                                AUTOdecoder_input, AUTOdecoder_hidden, encoded)
                        elif m_ver == 1:
                            AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                                AUTOdecoder_input, AUTOdecoder_hidden, di)
                        #print("di is", di)
                        #print("kl is ", kl)
                        #print("AUTOdecoder_output: " + str(AUTOdecoder_output.shape))
                        AUTOdecoder_input = AUTOtarget_tensor[:,di]  # Teacher forcing
                        di+=1
                        if di ==1:
                            tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])
                        else:
                            tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,  AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])], dim=1)
        
                else:
                    for di in range(AUTOtarget_tensor.shape[1]):
                        if m_ver == 0:
                                    AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                                        AUTOdecoder_input, AUTOdecoder_hidden, encoded)
                        elif m_ver == 1:
                            AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                                AUTOdecoder_input, AUTOdecoder_hidden, di)
                        topv, topi = AUTOdecoder_output.topk(1) 
                        AUTOdecoder_input = topi.squeeze().detach()  
                        di +=1
                        if di ==1:
                            tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])
                        else:
                            tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,  AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])], dim=1)
            
            else:
                for di in range(AUTOtarget_tensor.shape[1]): # for all sequences
                    """
                    decoder_in (Variable): [b]
                    encoded (Variable): [b x seq x hid]
                    input_out (np.array): [b x seq]
                    s (Variable): [b x hid]
                    """
                    #assert binary_vectors.size(1) == y.size(1)
                    # 1st state
                    if di==0:
                        Auto_out, s, w = decoder(input_idx=AUTOdecoder_input, encoded=encoded,
                                        encoded_idx=input_out, prev_state= s1,
                                        weighted=w, order=j, train_mode=True)
                       
                    else:
                        tmp_out, s, w = decoder(input_idx=AUTOdecoder_input, encoded=encoded,
                                        encoded_idx=input_out, prev_state= s,
                                        weighted=w, order=j, train_mode=True)
    
                        Auto_out = torch.cat([Auto_out,tmp_out],dim=1)
                        
                    if not(use_teacher_forcing):
                        AUTOdecoder_input = out[:,-1].max(1)[1].squeeze() # train with sequence outputs
                    else:
                        AUTOdecoder_input = y[:,j] # train with ground truth
                    #out_list.append(out[:,-1].max(1)[1].squeeze().cpu().data.numpy())
                tensor_of_all_AUTOdecoded_outputs = Auto_out
                    
            #print("shape:",tensor_of_all_AUTOdecoded_outputs.topk(1)[1].squeeze(2).shape)
            
            #tensor_of_all_AUTOdecoded_outputs
            
            target = y.view(-1)
            pad_out = out.view(-1,out.shape[2])
            if args.pad ==1 :
                pad_out = pad_out + 0.0001
            pad_out = torch.log(pad_out)
            
            review_sent = AUTOtarget_tensor.view(-1)
            
            #review_auto_out = tensor_of_all_AUTOdecoded_outputs.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2])
            #review_auto_out = torch.log(review_auto_out+0.0001)
            review_auto_out = tensor_of_all_AUTOdecoded_outputs.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2])
            
            if args.paragen_pretrain == 1:
                joint_loss = temp_loss_function(review_auto_out  ,review_sent)
                translation_loss, reconstruction_loss,joint_loss,kl  = joint_loss, joint_loss, joint_loss, 0
                avg_translation_loss += translation_loss; avg_reconstruction_loss += reconstruction_loss; avg_joint_loss += joint_loss; avg_kl += kl

            
            elif args.cos_only == 0:
                if kl_bool == 0:
                    kl = None
                    if args.auto_copy ==0:
                        #if m_ver ==1 or (m_ver==0 and args.q_to_p ==1):
                        #if m_ver==0 and args.q_to_p ==1:
                        if m_ver ==0:
                            review_auto_out = torch.softmax(tensor_of_all_AUTOdecoded_outputs.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2]), dim=1)
                        if args.pad == 1:
                            review_auto_out+0.0001
                        review_auto_out = torch.log(review_auto_out)
    
                        translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target, review_auto_out  ,review_sent) 
                        avg_translation_loss += translation_loss; avg_reconstruction_loss += reconstruction_loss; avg_joint_loss += joint_loss
                        avg_multi_loss += reconstruction_loss
                    else:
                        if args.pad ==1:
                            review_auto_out = review_auto_out+0.0001
                        review_auto_out =  torch.log(review_auto_out)   
                        translation_loss, reconstruction_loss, joint_loss = loss_function(0, pad_out, target, review_auto_out  ,review_sent) 
    
                elif kl_bool ==1:
                    review_auto_out =  torch.log(review_auto_out+0.0001)  
                    translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target, review_auto_out ,review_sent, kl_true=True) 
                    avg_translation_loss += translation_loss; avg_reconstruction_loss += reconstruction_loss; avg_joint_loss += joint_loss; avg_kl += kl
      
                
                if args.cos_obj == 1:
                    encoded_p, hidden_ec_p = encoder(sorted_p_vectors, sorted_p_lengths)
                    p_rep =torch.cat([encoded_p[:, -1, :hidden_dim], encoded_p[:, 0, hidden_dim:]  ], 1) 
                    #turn back
                    
                    p_rep = p_rep[original_relative_idx_sorted]
                    if args.cos_hinge ==0:
                        if args.proj_use == 0:
                            cos_loss = cos_loss_function(p_rep, s1, torch.tensor([1.0]*batch_size).cuda())
                        else:
                            encoder.make_proj_mat()
                            cos_loss = cos_loss_function(torch.mm(encoder.proj_mat, p_rep.transpose(0,1)).transpose(0,1), torch.mm(encoder.proj_mat, s1.transpose(0,1)).transpose(0,1), torch.tensor([1.0]*batch_size).cuda())
                    else:
                        return_sorted_neg_vectors,neg_original_relative_idx_sorted, sorted_neg_lengths = sample_negatives(sorted_idxes)
                        encoded_neg, hidden_ec_neg = encoder(return_sorted_neg_vectors, sorted_neg_lengths)
                        neg_rep = torch.cat([encoded_neg[:, -1, :hidden_dim], encoded_neg[:, 0, hidden_dim:]  ], 1) 
                        neg_rep = neg_rep[neg_original_relative_idx_sorted]
                        cos_loss = hinge_cos_loss(s1,p_rep,neg_rep ,args.margin)
                    avg_cos_loss += cos_loss
                    joint_loss += args.alpha* args.cos_alph* cos_loss
                
            elif args.cos_only == 1:
                encoded_p, hidden_ec_p = encoder(sorted_p_vectors, sorted_p_lengths)
                p_rep =torch.cat([encoded_p[:, -1, :hidden_dim], encoded_p[:, 0, hidden_dim:]  ], 1) 
                #turn back
                p_rep = p_rep[original_relative_idx_sorted]
                if args.proj_use == 0:
                    cos_loss = cos_loss_function(p_rep, s1, torch.tensor([1.0]*batch_size).cuda())
                else:
                    encoder.make_proj_mat()
                    cos_loss = cos_loss_function(torch.mm(encoder.proj_mat, p_rep.transpose(0,1)).transpose(0,1), torch.mm(encoder.proj_mat, s1.transpose(0,1)).transpose(0,1), torch.tensor([1.0]*batch_size).cuda())

                avg_cos_loss += cos_loss
                translation_loss = temp_loss_function(pad_out, target)
                joint_loss = translation_loss + args.alpha* args.cos_alph* cos_loss
                
            joint_loss.backward()
            
            
            if args.clip>0.0:                
                torch.nn.utils.clip_grad_norm(encoder.parameters(), args.clip)
                torch.nn.utils.clip_grad_norm(decoder.parameters(), args.clip)
                torch.nn.utils.clip_grad_norm(AUTOdecoder.parameters(), args.clip)
            if args.freeze_Enc ==0:
                opt_e.step()
            opt_d.step()
            if alpha !=0:
                opt_a.step()
            step +=1
            batch_num +=1
            batch_size = original_batch_size
                        
        if epoch%1 ==0:
            learning_rate= learning_rate * weight_decay
        
        if epoch %1 ==0:
            training_loss_dict['reconstruction'].append(avg_reconstruction_loss.item()/len(training_sampled))
            training_loss_dict['translation'].append(avg_translation_loss.item()/len(training_sampled))
            training_loss_dict['joint'].append(avg_joint_loss.item()/len(training_sampled))
            training_loss_dict['cos'].append(avg_cos_loss.item()/len(training_sampled))
            if kl_bool == 1:
                training_loss_dict['kl'].append(avg_kl.item()/len(training_sampled))

        
        elapsed = time.time()
        print("Elapsed time for epoch: ",elapsed-start)
        start = time.time()
    
        ##########VALIDATION####################################
        if epoch % 1 == 0 and args.skip_validation ==0:
            translation_pairs[epoch] = []; reconstruction_pairs[epoch] = []
            #current_loss_list = []
            test_batch_num = 0
            
            board_translation_loss, board_reconstruction_loss, board_joint_loss, board_kl = torch.tensor(0.0).cuda(),  torch.tensor(0.0).cuda(),  torch.tensor(0.0).cuda(),  torch.tensor(0.0).cuda()
            with torch.no_grad():
                while (test_batch_num) * batch_size < len(validation_sampled):
                    sentence_vectors, binary_vectors, target, X_lengths, rel_sorted_idxes = prepare_batch_test(test_batch_num)
                    batch_size = sentence_vectors.shape[0]
                    test_batch_num +=1
                    input_out = sentence_vectors.view(batch_size,-1).data.cpu().numpy()
                    
                    # mask input to remove padding
                    input_mask = np.array(input_out>0, dtype=int)
            
                    # input and output in Variable form
                    x = sentence_vectors
                    y = target.view(batch_size, -1)
            
                    # apply to encoder
                    if args.bert_sent == 0:
                        encoded, hidden_ec = encoder(x, X_lengths) # z is [batch_size, latent_size]
                    else:
                        encoded = torch.tensor(validation_encoded[rel_sorted_idxes][:, :max(X_lengths), :]).cuda()
                        hidden_ec = torch.tensor(validation_hidden[rel_sorted_idxes]).cuda()
            
                    # get initial input of decoder
                    decoder_in, s, w = decoder_initial(batch_size)
                    SOS_token = 1
                    decoder_in = torch.LongTensor([SOS_token] * batch_size)
                    if args.bert_sent ==0 and args.GRAN==0:
                        if args.bi==0:
                            s1 = encoded[:, -1, :] 
                        else:
                            s1 = torch.cat([encoded[:, -1, :hidden_dim], encoded[:, 0, hidden_dim:]  ], 1) # b x seq x 2*hidden 
                    else:
                        s1 = hidden_ec
                    #decoder_in = decoder_in.cpu()
                    if torch.cuda.is_available():
                        decoder_in = decoder_in.cuda()
            
                    # out_list to store outputs
                    out_list=[]
                    for j in range(y.size(1)): # for all sequences
                        """
                        decoder_in (Variable): [b]
                        encoded (Variable): [b x seq x hid]
                        input_out (np.array): [b x seq]
                        s (Variable): [b x hid]
                        """
                        # 1st state
                        if j==0:
                            out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                            encoded_idx=input_out, prev_state= s1,
                                            weighted=w, order=j, train_mode= False)
                        # remaining states
                        else:
                            tmp_out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                            encoded_idx=input_out, prev_state= s,
                                            weighted=w, order=j, train_mode= False)
                            out = torch.cat([out,tmp_out],dim=1)
                        decoder_in = out[:,-1].max(1)[1].squeeze() # train with sequence outputs
                        out_list.append(out[:,-1].max(1)[1].squeeze().cpu().data.numpy())
                    
                    
                    ##AUTO DECODER
                    ##
                    AUTOdecoder_hidden, AUTOdecoder_input, AUTOtarget_tensor, di = s1, torch.tensor([SOS_token]*batch_size, device=device), sentence_vectors, 1
                    use_teacher_forcing = False 
                    
                    if args.auto_copy == 0:
                        if use_teacher_forcing:
                            for di in range(AUTOtarget_tensor.shape[1]):
                                if m_ver == 0:
                                    AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                                        AUTOdecoder_input, AUTOdecoder_hidden, encoded)
                                elif m_ver == 1:
                                    AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                                        AUTOdecoder_input, AUTOdecoder_hidden, di)
                                AUTOdecoder_input = AUTOtarget_tensor[:,di]  # Teacher forcing
                                di+=1
                                if di ==1:
                                    tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])
                                else:
                                    tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,  AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])], dim=1)
                
                        else:
                            for di in range(AUTOtarget_tensor.shape[1]):
                                if m_ver == 0:
                                    AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                                        AUTOdecoder_input, AUTOdecoder_hidden, encoded)
                                elif m_ver == 1:
                                    AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                                        AUTOdecoder_input, AUTOdecoder_hidden, di)
                                topv, topi = AUTOdecoder_output.topk(1) 
                                AUTOdecoder_input = topi.squeeze().detach()  
                                di +=1
                                if di ==1:
                                    tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])
                                else:
                                    tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,  AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])], dim=1)
                    
                    else:
                        for di in range(AUTOtarget_tensor.shape[1]): # for all sequences
                            """
                            decoder_in (Variable): [b]
                            encoded (Variable): [b x seq x hid]
                            input_out (np.array): [b x seq]
                            s (Variable): [b x hid]
                            """
                            #assert binary_vectors.size(1) == y.size(1)
                            # 1st state
                            if di==0:
                                Auto_out, s, w = decoder(input_idx=AUTOdecoder_input, encoded=encoded,
                                                encoded_idx=input_out, prev_state= s1,
                                                weighted=w, order=j, train_mode=True)
                               
                            else:
                                tmp_out, s, w = decoder(input_idx=AUTOdecoder_input, encoded=encoded,
                                                encoded_idx=input_out, prev_state= s,
                                                weighted=w, order=j, train_mode=True)
            
                                Auto_out = torch.cat([Auto_out,tmp_out],dim=1)
                                
                            if not(use_teacher_forcing):
                                AUTOdecoder_input = out[:,-1].max(1)[1].squeeze() # train with sequence outputs
                            else:
                                AUTOdecoder_input = y[:,j] # train with ground truth
                            #out_list.append(out[:,-1].max(1)[1].squeeze().cpu().data.numpy())
                        tensor_of_all_AUTOdecoded_outputs = Auto_out
                            
  
                    
                    target = y.view(-1)
                    pad_out = out.view(-1,out.shape[2])
                    pad_out = pad_out + 0.0001
                    pad_out = torch.log(pad_out)
                    
                    review_sent = sentence_vectors.view(-1)
                    review_auto_out = tensor_of_all_AUTOdecoded_outputs.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2])
                    review_auto_out =  torch.log(review_auto_out + 0.0001)
                                        
                    if kl_bool == 0:
                        kl = None
                        translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target, review_auto_out  ,review_sent, kl_true=False) 
                        board_translation_loss += translation_loss; board_reconstruction_loss += reconstruction_loss; board_joint_loss += joint_loss

                    elif kl_bool ==1:
                        translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target, review_auto_out ,review_sent, kl_true=True) 
                        board_translation_loss += translation_loss; board_reconstruction_loss += reconstruction_loss; board_joint_loss += joint_loss; board_kl += kl

               
                    predicted = np.transpose(np.array(out_list))    
                    translation_pairs[epoch].append((predicted, y.cpu().numpy()))
                    reconstruction_pairs[epoch].append((tensor_of_all_AUTOdecoded_outputs.topk(1)[1].squeeze(2).cpu().numpy(), sentence_vectors.cpu().numpy()))
                    batch_size = original_batch_size
                    
                                        
            
            
            for i in range(y.shape[0]):
                print("target: " +str(' '.join([LFidx2vocab[y[i][j].item()] for j in range(y.shape[1]) if y[i][j].item() != 0])))
                temp_list = []
                for j in range(predicted.shape[1]):
                    temp_list.append(LFidx2vocab[predicted[i][j]])
                #print("predicted: " + str([LFidx2vocab[out_list[i].item()] for i in range(len(out_list))]))
                print("predicted: " + str(' '.join(temp_list)))
                print(' ')
                    
            
            
            for i in range(sentence_vectors.shape[0]):
                print("target: " +str(' '.join([LFidx2vocab[sentence_vectors[i][j].item()] for j in range(sentence_vectors.shape[1]) if sentence_vectors[i][j].item() != 0])))
                temp_list = []
                for j in range(tensor_of_all_AUTOdecoded_outputs.topk(1)[1].squeeze(2).cpu().numpy().shape[1]):
                    temp_list.append(LFidx2vocab[tensor_of_all_AUTOdecoded_outputs.topk(1)[1].cpu().numpy().squeeze(2)[i][j]])
                #print("predicted: " + str([LFidx2vocab[out_list[i].item()] for i in range(len(out_list))]))
                print("predicted: " + str(' '.join(temp_list)))
                print(' ')
           
            temp_trans_pairs = {epoch: translation_pairs[epoch]}
            temporary_dict = {'translation_pairs':temp_trans_pairs}
            
            if args.cos_obj == 1 and args.cos_only ==0:
                print("Cos Loss: ", avg_cos_loss.item()/len(training_sampled))
                print("Multi Loss: ", avg_multi_loss.item()/len(training_sampled))
                
            print("Training Translation loss: ", avg_translation_loss.item()/len(training_sampled))
            print("Training Reconstruction loss: ", avg_reconstruction_loss.item()/len(training_sampled))
                        
            print("VALIDATION ACCURACY FOR EPOCH ", epoch+1, " : ", acc_bowser(temporary_dict))
            
            temp_trans_pairs = {epoch: reconstruction_pairs[epoch]}
            temporary_dict = {'translation_pairs':temp_trans_pairs}
            
            print("Reconstruction ACCURACY FOR EPOCH ", epoch+1, " : ", acc_bowser(temporary_dict))
            #print("jjamppong")

            validation_loss_dict['reconstruction'].append(board_reconstruction_loss.item()/len(validation_sampled))
            validation_loss_dict['translation'].append(board_translation_loss.item()/len(validation_sampled))
            validation_loss_dict['joint'].append(board_joint_loss.item()/len(validation_sampled))
            if kl_bool ==1:
                validation_loss_dict['kl'].append(board_kl.item()/len(len(validation_sampled)))
                    
                #test_loss_list.append(current_loss_list)
        if args.save_every_epoch ==1 and epoch>= args.save_from:
            save_every_epoch(translation_pairs, reconstruction_pairs, args, epoch)
        save(translation_pairs, reconstruction_pairs, args)
        log(args, epoch)
    
    #writer.close()
    

if __name__ == '__main__':
    print("running")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.cuda is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)

    num_epochs = 25
    if args.cos_only == 15:
        num_epochs = 16
    train(num_epochs, args)
    
    
