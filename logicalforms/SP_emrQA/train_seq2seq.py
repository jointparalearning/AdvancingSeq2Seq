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
from models.seq2seq_Luong import *
from models.plain_method2 import *
from models.masked_cross_enropy import *


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
parser.add_argument('-copy', '--copy', type=int, metavar='', required=True, help="saving directory of output file; not required but can specify if you want to.")
parser.add_argument('-which_attn', '--which_attn', type = str,metavar='', required=True )
parser.add_argument('-bi', '--bi', type = int,metavar='', required=True )
parser.add_argument('-qp', '--q_to_p', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-wl', '--which_loss', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-s_e_e', '--save_every_epoch', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-s_from', '--save_from', type = int, metavar='', default = 69, help ="conditional copy")
parser.add_argument('-cos_obj', '--cos_obj', type = int, metavar='', default = 0, help ="conditional copy")
parser.add_argument('-cos_alph', '--cos_alph', type = float, metavar='', default = 1.0, help ="conditional copy")
parser.add_argument('-cos_only', '--cos_only', type = int, metavar='', default = 0)

##Optional Arguments
parser.add_argument('-H', '--hyperparams', type=str, metavar='', required=False, help = "hyperparameters")
parser.add_argument('-load_dir', '--loading_dir', type=str, metavar='', required=False, help="load model and start with it if there is a directory")
parser.add_argument('-c', '--cuda', type = int, metavar='', required=False, help = "cuda")

parser.add_argument('-s', '--seed', type = float, metavar='', default=1, help = "cuda")
parser.add_argument('-av', '--auto_vae', type = int, metavar = '', default = 0)
parser.add_argument('-as', '--auto_simple', type = int, metavar = '', default = 0)


args = parser.parse_args()

m_ver_dict = {0: 'models/plain.py', 1:'models/vae_ver1.py', 2:  'models/plain_explicit.py', 3:'models/vae_explicit.py'}

torch.manual_seed(1000*args.seed)


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

def train(num_epochs, args):
    #if args.q_to_p == 1:
    #    num_epochs = 150
    def log(args, epoch):
        file_name = "outputs/" + args.saving_dir + "/logs.txt"
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
        torch.save(encoder.state_dict(),'outputs/' + args.saving_dir + '/encoder.pt')
        torch.save(decoder.state_dict(),'outputs/' + args.saving_dir + '/decoder.pt')
        torch.save(AUTOdecoder.state_dict(),'outputs/' + args.saving_dir + '/auto-decoder.pt')
        pairs_dict = {'translation_pairs': trans_pairs, 'reconstruction_pairs': rec_pairs}
        loss_dict = {'training':training_loss_dict, 'validation': validation_loss_dict}
        final_dict = {'pairs_dict':pairs_dict, 'loss_dict':loss_dict}
        cPickle.dump(final_dict, open('outputs/' + args.saving_dir + '/loss_list.p',"wb"))    
       
    def save_every_epoch(trans_pairs, rec_pairs, args, epoch):
        torch.save(encoder.state_dict(),'outputs/' + args.saving_dir + '/encoder_epoch' + str(epoch) +  '.pt')
        torch.save(decoder.state_dict(),'outputs/' + args.saving_dir + '/decoder_epoch' + str(epoch) +  '.pt')
        torch.save(AUTOdecoder.state_dict(),'outputs/' + args.saving_dir + '/auto-decoder_epoch' + str(epoch) +  '.pt')
        pairs_dict = {'translation_pairs': trans_pairs, 'reconstruction_pairs': rec_pairs}
        loss_dict = {'training':training_loss_dict, 'validation': validation_loss_dict}
        final_dict = {'pairs_dict':pairs_dict, 'loss_dict':loss_dict}
        cPickle.dump(final_dict, open('outputs/' + args.saving_dir + '/loss_list.p',"wb"))    
        

        
    def prepare_batch_training(batch_num):
        if (batch_num+1)*batch_size <= len(training_sampled):
            end_num = (batch_num+1)*batch_size 
        else:
            end_num = len(training_sampled)
            #batch_size = end_num - batch_num*batch_size
        #sort by length
        sorted_idxes = sorted(training_sampled[batch_num*batch_size:end_num], key = lambda idx: len(sent2idxtensor(tokenized_eng_sentences[idx], idx)), reverse=True)
        #print("sorted idxes is " + str(sorted_idxes))
        max_length = len(sent2idxtensor(tokenized_eng_sentences[sorted_idxes[0]], sorted_idxes[0]))
        max_y_length = max([len(Qidx2LFIdxVec_dict[idx]) for idx in sorted_idxes])
        #print("max length is " + str(max_length))
        sentence_vectors, x_lengths, labels, targ_lengths = [],[],[],[]
        for idx in sorted_idxes:
            sent = sent2idxtensor(tokenized_eng_sentences[idx], idx)
            sentence_vectors.append(sent +[pad_token] * (max_length-len(sent)))
            x_lengths.append(len(sent))
            labels.append(Qidx2LFIdxVec_dict[idx] +[pad_token] * (max_y_length - len(Qidx2LFIdxVec_dict[idx])) )
            targ_lengths.append(len(Qidx2LFIdxVec_dict[idx]))
        return torch.tensor(sentence_vectors, device=device), torch.tensor(labels, device=device), x_lengths, targ_lengths, sorted_idxes, max_y_length
                
    def prepare_batch_test(test_batch_num):
        if (test_batch_num+1)*batch_size <= len(validation_sampled):
            end_num = (test_batch_num+1)*batch_size 
        else:
            end_num = len(validation_sampled)
            #batch_size = end_num - batch_num*batch_size
        sorted_idxes = sorted(validation_sampled[test_batch_num*batch_size:end_num], key = lambda idx: len(sent2idxtensor(tokenized_eng_sentences[idx], idx)), reverse=True)
        #print("sorted idxes:" + str(sorted_idxes))
        max_length = len(sent2idxtensor(tokenized_eng_sentences[sorted_idxes[0]], sorted_idxes[0]))
        max_y_length = max([len(Qidx2LFIdxVec_dict[idx]) for idx in sorted_idxes])
        sentence_vectors, x_lengths, labels, targ_lengths = [],[],[],[]
        for idx in sorted_idxes:
            sent = sent2idxtensor(tokenized_eng_sentences[idx], idx)
            sentence_vectors.append(sent +[pad_token] * (max_length-len(sent)))
            x_lengths.append(len(sent))
            labels.append(Qidx2LFIdxVec_dict[idx] +[pad_token] * (max_y_length - len(Qidx2LFIdxVec_dict[idx])) )
            targ_lengths.append(len(Qidx2LFIdxVec_dict[idx]))
        return torch.tensor(sentence_vectors, device=device), torch.tensor(labels, device=device), x_lengths, targ_lengths, max_y_length
    
    def prepare_q_to_p_training(batch_num):
        #TODO HERE
        p_vectors = []
        p_idxes = []
        SOS_token = 1; EOS_token = 2

        #valid_count = 0
        for idx in sorted_idxes:
            if q_to_p[idx] != []:
                valid_ps = [p for p in q_to_p[idx] if p in seventy_percent_idx_dict]
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
        p_lengths = []
        temp_p_vectors = []
        for p_idx in p_idxes:
            p_vector = [SOS_token] + [Qvocab2idx[token] for token in tokenized_eng_sentences[p_idx[0]]]+[EOS_token]
            temp_p_vectors.append(p_vector)
            p_lengths.append(len(p_vector))
            p_vector = p_vector + [pad_token] * (max_p_length - len(p_vector))
            p_vectors.append(p_vector)
        
        return_p_idxes = [p_idx[0] for p_idx in p_idxes]
        
        sorted_p_vectors = sorted(temp_p_vectors, key=len, reverse=True)
        original_relative_idx_sorted = sorted(range(len(temp_p_vectors)), key=lambda k: temp_p_vectors[k], reverse=True)
        sorted_back_to_original_dict = {idx:i for i,idx in enumerate(original_relative_idx_sorted)}
        sorted_back_to_original_list = [-1] * len(p_idxes)
        for idx, i in sorted_back_to_original_dict.items():
            sorted_back_to_original_list[idx] = i
        sorted_p_lengths = [len(sorted_l) for sorted_l in sorted_p_vectors]
        return_sorted_p_vectors = [sorted_inp + [pad_token] * (max_p_length - len(sorted_inp))for sorted_inp in sorted_p_vectors]

        
        #print("non-auto percent:" , valid_count/batch_size)
        return torch.tensor(p_vectors, device = device), return_p_idxes, p_lengths, max_p_length, torch.tensor(return_sorted_p_vectors, device = device),sorted_back_to_original_list, sorted_p_lengths



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
    
    #writer = SummaryWriter()
    global shuffle_scheme
    global split_num
    cuda, m_ver, alpha, shuffle_scheme, kl_bool, save_dir, load_dir= args.cuda, args.model_version, args.alpha, args.shuffle_scheme, args.loss_version, args.saving_dir, args.loading_dir
        
    #assert args.split_num in [-2, -1, 0, 1,2,3,4,5]
    split_num = args.split_num
        
    ###make folder for save_dir
    save_dir = "outputs/" + save_dir 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    ###load the required arguments first
    #load shuff_scheme
    if shuffle_scheme in [0,1,2]:
        exec(open('data_prep/data_prepRAW_Shuffle.py').read(), globals(), globals())        
    else:
        raise ValueError('shuffling scheme argument wrong')
        
    if args.q_to_p is 1:
        q_to_p = cPickle.load(open('data/q_to_p_dict.p','rb'))

    
    #load alpha     
    alpha = alpha 
    
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
    kl = None
        
    if args.hyperparams is not None:
        exec(open('hyperparams/'+args.hyperparams).read(), globals(), globals())
    else:
        #exec(compile(open(file).read(), file, 'exec'))
        exec(open('hyperparams/default_hyperparams.py').read(), globals(), globals()) 
    original_batch_size = batch_size
    
    #load model version
    #assert m_ver in m_ver_dict
    #exec(open(m_ver_dict[m_ver]).read(), globals(), globals())
    #initialize model and load if needed 
    if m_ver in [0,2]:
        if args.copy is 0:
            encoder = EncoderRNN(vocab_size, hidden_dim)
            decoder = LuongAttnDecoderRNN('general',hidden_dim, vocab_size, bi=args.bi)
            if args.auto_vae == 0:
                AUTOdecoder = LuongAttnDecoderRNN('general',hidden_dim, vocab_size, bi=args.bi)
            else:
                AUTOdecoder = VAESeq2seqDecoder(AUTOhidden_dim, vocab_size, latent_dim, bi=args.bi)
            #decoder = LuongAttnDecoderRNN(args.which_attn,hidden_dim, vocab_size)   
            #decoder = BahdanauAttnDecoderRNN('concat',hidden_dim, vocab_size)   
        elif args.copy is 1:
            encoder = CopyEncoderRAW(hidden_dim, vocab_size)
            decoder = CopyDecoder(vocab_size, embed_dim, hidden_dim)

    elif m_ver in [1,3]:
        raise NotImplementedError
                
    else:
        raise NotImplementedError

    print("worked before cuda encoder")
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        AUTOdecoder.cuda()

    #load save_dir
    
    #load loss function
    loss_function = joint_loss_nll  
    temp_loss_function = nn.NLLLoss(ignore_index = pad_token)
        
    #load model 
    if load_dir:
        encoder_dir = 'outputs/' + load_dir + '/encoder.pt'  
        decoder_dir = 'outputs/' + load_dir + '/decoder.pt'
        auto_dir = 'outputs/' + load_dir + '/auto-decoder.pt'
        
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
        trained_until = 182
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
    cos_loss_function = torch.nn.CosineEmbeddingLoss()

    
    if kl_bool ==1:
        training_loss_dict['kl'] = []
        validation_loss_dict['kl'] = []
    start = time.time()
    step = 0
    for epoch in epochs:
        
        opt_e = optim.Adam(params=encoder.parameters(),  lr=learning_rate)
        opt_d = optim.Adam(params=decoder.parameters(),  lr=learning_rate)
        opt_a = optim.Adam(params=AUTOdecoder.parameters(), lr=learning_rate)

                
        batch_num = 0         
        current_loss_list = []
        print("==================================================")
        print("Epoch ",epoch)
        board_translation_loss, board_reconstruction_loss, board_joint_loss, board_kl, board_cos_loss = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        while (batch_num) * batch_size < len(training_sampled):
            if batch_num % 100 ==0:
                print("==================================================")
                print("Batch Num: ",batch_num)
                print("Batch Percent: ",100 *(batch_num) * batch_size/ len(training_sampled), "%")
            sentence_vectors, target, X_lengths, targ_lengths, sorted_idxes, max_y_length = prepare_batch_training(batch_num)
            if args.q_to_p is 1:
                p_vectors, p_idxes, p_lengths, max_p_length, sorted_p_vectors,original_relative_idx_sorted, sorted_p_lengths = prepare_q_to_p_training(batch_num)

            
            #batch_size = sentence_vectors.shape[0]
            batch_size = sentence_vectors.shape[0]
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
            encoded, hidden_ec = encoder(x, X_lengths) # z is [batch_size, latent_size]
            #print("encoded shape", encoded.shape) #should be [Batch x Seq x Hidden]
            if math.isnan(encoded[0][0].data[0]):
                print("encoder broken!")
                sys.exit()
                break
            
            
            #FROM HERE
            # get initial input of decoder
            SOS_token = 1
            decoder_input = torch.LongTensor([SOS_token] * batch_size)
            #print("n layers is: ", decoder.n_layers)
            #print("output size is: ", decoder.output_size)
            #print("X lengths are:", X_lengths)
            if args.bi ==0:
                s1 = encoded[:, -1, :]  # Use last (forward) hidden state from encoder
            else:
                s1 = torch.cat([encoded[:, -1, :hidden_dim], encoded[:, 0, hidden_dim:]  ], 1)
            decoder_hidden=s1
        #print("decodedr hidden shape", decoder_hidden.shape) #should be 1x Batch x Hidden
            
            # We will flip because we have the batch first
            try:
                all_decoder_outputs = torch.zeros(batch_size, max_y_length, decoder.output_size)
            except:
                all_decoder_outputs = torch.zeros(batch_size, max_y_length, decoder.vocab_size)
            #print("all_decoder_outputs shape", all_decoder_outputs.shape)
            # Shape: [5, 20, 298]
            decoder_hidden = s1
            if USE_CUDA:
                all_decoder_outputs = all_decoder_outputs.cuda()
                decoder_input = decoder_input.cuda()

            # Run through decoder one time step at a time
            if args.copy is 0:
                for t in range(max_y_length):
                    decoder_output, decoder_hidden, decoder_attn = decoder(
                        decoder_input, decoder_hidden, encoded
                    )
                    #print("decoder output shape", decoder_output.shape)
                    #print("outputted decoder hidden shape", decoder_hidden.shape)
                    #print("decoder_attn shape", decoder_attn.shape)
                    #print("shape of decoder output", decoder_output.shape)
                    if t ==0:
                        out = decoder_output
                    else:
                        out = torch.cat([out,decoder_output],dim=1)
                    all_decoder_outputs[:, t] = decoder_output  # Store this step's outputs [5, 298]
                    decoder_input = target[:, t]  # Next input is current target
    
                    # all_decoder_outputs.topk(1)[0].size() - [5,48,1]
                    # For reference the topk 1, refers to 1 from the top. It always does it over the last dimension
                    # (vocab_size) and keep the other dimensions as it is
            
            
                out = all_decoder_outputs
                if args.q_to_p == 0:
                    AUTOdecoder_hidden, AUTOdecoder_input, AUTOtarget_tensor, di = s1, torch.tensor([SOS_token]*batch_size, device=device), sentence_vectors, 1
                else:
                    AUTOdecoder_hidden, AUTOdecoder_input, AUTOtarget_tensor, di = s1, torch.tensor([SOS_token]*batch_size, device=device), p_vectors, 1
                use_teacher_forcing = True if random.random() < teacher_forcing_prob else False
                
                if use_teacher_forcing:
                    for di in range(AUTOtarget_tensor.shape[1]):
                        if args.auto_vae==0:
                                AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                                    AUTOdecoder_input, AUTOdecoder_hidden, encoded)
                        else:
                            AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                                    AUTOdecoder_input, AUTOdecoder_hidden, di, batch_size)
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
                        if args.auto_vae==0:
                                AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                                    AUTOdecoder_input, AUTOdecoder_hidden, encoded)
                        else:
                            AUTOdecoder_output, AUTOdecoder_hidden, kl = AUTOdecoder(
                                    AUTOdecoder_input, AUTOdecoder_hidden, di,  batch_size)
                        topv, topi = AUTOdecoder_output.topk(1) 
                        AUTOdecoder_input = topi.squeeze().detach()  
                        di +=1
                        if di ==1:
                            tensor_of_all_AUTOdecoded_outputs = AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])
                        else:
                            tensor_of_all_AUTOdecoded_outputs = torch.cat([tensor_of_all_AUTOdecoded_outputs,  AUTOdecoder_output.view(AUTOdecoder_output.shape[0],1,AUTOdecoder_output.shape[1])], dim=1)
        
    
            
            elif args.copy is 1:
                use_teacher_forcing = True
                s = decoder_hidden
                w = None
                for j in range(y.size(1)): # for all sequences
                    """
                    decoder_in (Variable): [b]
                    encoded (Variable): [b x seq x hid]
                    input_out (np.array): [b x seq]
                    s (Variable): [b x hid]
                    """
                    #assert binary_vectors.size(1) == y.size(1)
                    # 1st state
                    if j==0:
                        out, s, w = decoder(input_idx=decoder_input, encoded=encoded,
                                        encoded_idx=input_out, prev_state= s,
                                        weighted=w, order=j, train_mode=True)
                       
                    else:
                        tmp_out, s, w = decoder(input_idx=decoder_input, encoded=encoded,
                                        encoded_idx=input_out, prev_state= s,
                                        weighted=w, order=j, train_mode=True)
    
                        out = torch.cat([out,tmp_out],dim=1)
                        
                    if not(use_teacher_forcing):
                        decoder_input = out[:,-1].max(1)[1].squeeze() # train with sequence outputs
                    else:
                        decoder_input = y[:,j] # train with ground truth
                all_decoder_outputs = out
            # print("Reaching here!")

            # Test masked cross entropy loss
            #print("all decoder outputs: ", all_decoder_outputs.shape)
            #print("target: ", target.shape)
            #print("targ lengths: ", targ_lengths)
            
            
            review_sent = AUTOtarget_tensor.view(-1)
            review_auto_out = tensor_of_all_AUTOdecoded_outputs.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2])


            if args.copy is 0 and args.cos_only ==0:
                if args.which_loss == 0:
                    target = y.view(-1)
                    pad_out = F.softmax(out, dim=2)
                    pad_out = pad_out.view(-1,pad_out.shape[2])
                    pad_out = torch.log(pad_out + 0.0001)
                    
                    review_auto_out = F.softmax(tensor_of_all_AUTOdecoded_outputs, dim=2)
                    review_auto_out= review_auto_out.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2])
                    review_auto_out = torch.log(review_auto_out + 0.0001)
                    
                    #review auto should do log softmax too 
                
                    kl = None
                    translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target, review_auto_out  ,review_sent) 
                    board_translation_loss += translation_loss; board_reconstruction_loss += reconstruction_loss; board_joint_loss += joint_loss
                    loss= joint_loss
                    #loss = temp_loss_function(pad_out, target)
                else:
                    loss = masked_cross_entropy(
                        (out).contiguous(),
                        target.contiguous(),
                        targ_lengths
                    )
                    if args.q_to_p ==0:
                         auto_loss = masked_cross_entropy(
                            (tensor_of_all_AUTOdecoded_outputs).contiguous(),
                            review_sent.contiguous(),
                            X_lengths
                        )
                    else:
                        auto_loss = masked_cross_entropy(
                            (tensor_of_all_AUTOdecoded_outputs).contiguous(),
                            review_sent.contiguous(),
                            p_lengths
                        )
                        loss = loss+alpha * auto_loss
                
            elif args.copy is 1 and args.cos_only ==0:
                target = y.view(-1)
                pad_out = out.view(-1,out.shape[2])
                pad_out = pad_out + 0.0001
                pad_out = torch.log(pad_out)
                loss = temp_loss_function(pad_out, target)
# =============================================================================
#                 loss = masked_cross_entropy(
#                     all_decoder_outputs.contiguous(),
#                     target.contiguous(),
#                     targ_lengths,
#                     True
#                 )
# =============================================================================
            # print('loss in batch %d:'%i, loss.item())
            #TO HERE
            
            if args.cos_obj == 1 and args.cos_only ==0:
                encoded_p, hidden_ec_p = encoder(sorted_p_vectors, sorted_p_lengths)
                p_rep =torch.cat([encoded_p[:, -1, :hidden_dim], encoded_p[:, 0, hidden_dim:]  ], 1) 
                #turn back
                p_rep = p_rep[original_relative_idx_sorted]
                cos_loss = cos_loss_function(p_rep, s1, torch.tensor([1.0]*batch_size).cuda())
                board_cos_loss += cos_loss
                loss += args.alpha* args.cos_alph* cos_loss
              
            if args.cos_only == 1:
                encoded_p, hidden_ec_p = encoder(sorted_p_vectors, sorted_p_lengths)
                p_rep =torch.cat([encoded_p[:, -1, :hidden_dim], encoded_p[:, 0, hidden_dim:]  ], 1) 
                #turn back
                p_rep = p_rep[original_relative_idx_sorted]
                cos_loss = cos_loss_function(p_rep, s1, torch.tensor([1.0]*batch_size).cuda())
                board_cos_loss += cos_loss
                
                
                
                target = y.view(-1)
                pad_out = F.softmax(out, dim=2)
                pad_out = pad_out.view(-1,pad_out.shape[2])
                pad_out = torch.log(pad_out + 0.0001)
                
                review_auto_out = F.softmax(tensor_of_all_AUTOdecoded_outputs, dim=2)
                review_auto_out= review_auto_out.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2])
                review_auto_out = torch.log(review_auto_out + 0.0001)
                
                #review auto should do log softmax too 
            
                kl = None
                translation_loss = temp_loss_function(pad_out, target)
                loss = translation_loss + args.alpha* args.cos_alph* cos_loss
            
            loss.backward()

            opt_e.step()
            opt_d.step()
            opt_a.step()
            
            target = y.view(-1)
# =============================================================================
#             pad_out = decoder_output.view(-1,decoder_output.shape[2])
#             pad_out = pad_out + 0.0001
#             pad_out = torch.log(pad_out)
#             
#             
# =============================================================================
# =============================================================================
#             if kl_bool == 0:
#                 kl = None
#                 translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target, review_auto_out  ,review_sent) 
#                 board_translation_loss += translation_loss; board_reconstruction_loss += reconstruction_loss; board_joint_loss += joint_loss
# 
#             elif kl_bool ==1:
#                 translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target, review_auto_out ,review_sent, kl_true=True) 
#                 board_translation_loss += translation_loss; board_reconstruction_loss += reconstruction_loss; board_joint_loss += joint_loss; board_kl += kl
#   
# =============================================================================
            
# =============================================================================
#             if len(pad_out[pad_out == -inf]) > 0 :
#                 joint_loss = torch.tensor(0.0, device=device)
#                 print("Loss broken but skipped!")
#             else:
#                 joint_loss.backward()
# =============================================================================
            

# =============================================================================
#             opt_e.step()
#             opt_d.step()
# =============================================================================
# =============================================================================
#             if alpha !=0:
#                 opt_a.step()
# =============================================================================
            step +=1
            batch_num +=1
            batch_size = original_batch_size
                        
        if epoch%1 ==0:
            learning_rate= learning_rate * weight_decay
        
        training_loss_dict['reconstruction'].append(board_reconstruction_loss.item()/len(training_sampled))
        training_loss_dict['translation'].append(board_translation_loss.item()/len(training_sampled))
        training_loss_dict['joint'].append(board_joint_loss.item()/len(training_sampled))
        training_loss_dict['cos'].append(board_cos_loss.item()/len(training_sampled))
        if kl_bool == 1:
            training_loss_dict['kl'].append(board_kl.item()/len(training_sampled))

        
        elapsed = time.time()
        print("Elapsed time for epoch: ",elapsed-start)
        start = time.time()
    
        ##########VALIDATION####################################
        if epoch % 1 == 0:
            translation_pairs[epoch] = []; reconstruction_pairs[epoch] = []
            #current_loss_list = []
            test_batch_num = 0
            
            board_translation_loss, board_reconstruction_loss, board_joint_loss, board_kl = 0, 0, 0, 0
            with torch.no_grad():
                while (test_batch_num) * batch_size < len(validation_sampled):
                    sentence_vectors, target, X_lengths, targ_lengths, max_y_length = prepare_batch_test(test_batch_num)
                    batch_size = sentence_vectors.shape[0]
                    test_batch_num +=1
                    input_out = sentence_vectors.view(batch_size,-1).data.cpu().numpy()
                    
                    # mask input to remove padding
                    input_mask = np.array(input_out>0, dtype=int)
            
                    # input and output in Variable form
                    x = sentence_vectors
                    y = target.view(batch_size, -1)
            
                    # apply to encoder
                    encoded, hidden_ec = encoder(x, X_lengths) # z is [batch_size, latent_size]
                    
                
                    # get initial input of decoder
                    SOS_token = 1
                    decoder_input = torch.LongTensor([SOS_token] * batch_size)
                    if args.bi ==0:
                        s1 = encoded[:, -1, :]  # Use last (forward) hidden state from encoder
                    else:
                        s1 = torch.cat([encoded[:, 0, :hidden_dim] , encoded[:, -1, hidden_dim:]  ], 1)
                    decoder_hidden=s1
                    #print("decodedr hidden shape", decoder_hidden.shape) #should be 1x Batch x Hidden
                    
                    # We will flip because we have the batch first
                    try:
                        all_decoder_outputs = torch.zeros(batch_size, max_y_length, decoder.output_size)
                    except:
                        all_decoder_outputs = torch.zeros(batch_size, max_y_length, decoder.vocab_size)
                    # We will flip because we have the batch first
                    # Shape: [5, 20, 298]
        
                    if USE_CUDA:
                        all_decoder_outputs = all_decoder_outputs.cuda()
                        decoder_input = decoder_input.cuda()
                        
                    
                    out_list = []
                    rec_out_list = []
                    if args.copy is 0:
                        for t in range(max_y_length):
                            decoder_output, decoder_hidden, decoder_attn = decoder(
                                decoder_input, decoder_hidden, encoded
                            )
                            #print("decoder output shape", decoder_output.shape)
                            #print("outputted decoder hidden shape", decoder_hidden.shape)
                            #print("decoder_attn shape", decoder_attn.shape)
                            #print("shape of decoder output", decoder_output.shape)
                            out_list.append(decoder_output[:].max(1)[1].squeeze().cpu().data.numpy())
                            all_decoder_outputs[:, t] = decoder_output  # Store this step's outputs [5, 298]
                            decoder_input = decoder_output.max(1)[1] #no teacher forcing
                            # all_decoder_outputs.topk(1)[0].size() - [5,48,1]
                            # For reference the topk 1, refers to 1 from the top. It always does it over the last dimension
                            # (vocab_size) and keep the other dimensions as it is
                    
                        AUTOdecoder_hidden, AUTOdecoder_input, AUTOtarget_tensor, di = s1, torch.tensor([SOS_token]*batch_size, device=device), sentence_vectors, 1
                        use_teacher_forcing = False 
                        
                        if use_teacher_forcing:
                            for di in range(AUTOtarget_tensor.shape[1]):
                                if args.auto_vae==0:
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
                                if args.auto_vae==0:
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
                    
                    
                    elif args.copy is 1:
                        use_teacher_forcing = False
                        s = decoder_hidden
                        w = None
                        for j in range(y.size(1)): # for all sequences
                            """
                            decoder_in (Variable): [b]
                            encoded (Variable): [b x seq x hid]
                            input_out (np.array): [b x seq]
                            s (Variable): [b x hid]
                            """
                            #assert binary_vectors.size(1) == y.size(1)
                            # 1st state
                            if j==0:
                                out, s, w = decoder(input_idx=decoder_input, encoded=encoded,
                                                encoded_idx=input_out, prev_state= s,
                                                weighted=w, order=j, train_mode=True)
                               
                            else:
                                tmp_out, s, w = decoder(input_idx=decoder_input, encoded=encoded,
                                                encoded_idx=input_out, prev_state= s,
                                                weighted=w, order=j, train_mode=True)
            
                                out = torch.cat([out,tmp_out],dim=1)
                                
                            if not(use_teacher_forcing):
                                decoder_input = out[:,-1].max(1)[1].squeeze() # train with sequence outputs
                            else:
                                decoder_input = y[:,j] # train with ground truth
                            out_list.append(out[:,-1].max(1)[1].squeeze().cpu().data.numpy())
                        all_decoder_outputs = out
                    
                    
                    
                    # all_decoder_outputs.topk(1)[0].size() - [5,48,1]
                    # For reference the topk 1, refers to 1 from the top. It always does it over the last dimension
                    # (vocab_size) and keep the other dimensions as it is
                    out = all_decoder_outputs
                    review_sent = AUTOtarget_tensor.view(-1)
                    review_auto_out = tensor_of_all_AUTOdecoded_outputs.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2])

                    
                    target = y.view(-1)
                    pad_out = F.softmax(out, dim=2)
                    pad_out = pad_out.view(-1,pad_out.shape[2])
                    pad_out = torch.log(pad_out + 0.0001)
                    
                    review_auto_out = F.softmax(tensor_of_all_AUTOdecoded_outputs, dim=2)
                    review_auto_out= review_auto_out.view(-1, tensor_of_all_AUTOdecoded_outputs.shape[2])
                    review_auto_out = torch.log(review_auto_out + 0.0001)
                    
                    #review auto should do log softmax too 
                
                    kl = None
                    translation_loss, reconstruction_loss, joint_loss = loss_function(alpha, pad_out, target, review_auto_out  ,review_sent) 
                    board_translation_loss += translation_loss; board_reconstruction_loss += reconstruction_loss; board_joint_loss += joint_loss

                 
                    predicted = np.transpose(np.array(out_list)) 
                    #print("out shape: ", out_list[0].shape)
                    predicted_auto = np.transpose(np.array(rec_out_list))
                    #print("rec out shape: ", rec_out_list[0].shape)
                    #print("predicted shape", predicted.shape)
                    translation_pairs[epoch].append((predicted, y.cpu().numpy()))
                    reconstruction_pairs[epoch].append((predicted_auto, x.cpu().numpy()))
                    batch_size = original_batch_size
                    
                                        
                temp_trans_pairs = {epoch: translation_pairs[epoch]}
                temporary_dict = {'translation_pairs':temp_trans_pairs}
                
                for i in range(y.shape[0]):
                    print("target: " +str(' '.join([LFidx2vocab[y[i][j].item()] for j in range(y.shape[1]) if y[i][j].item() != 0])))
                    temp_list = []
                    for j in range(predicted.shape[1]):
                        temp_list.append(LFidx2vocab[predicted[i][j]])
                    #print("predicted: " + str([LFidx2vocab[out_list[i].item()] for i in range(len(out_list))]))
                    print("predicted: " + str(' '.join(temp_list)))
                    print(' ')
                    
                print("VALIDATION ACCURACY FOR EPOCH ", epoch+1, " : ", acc_bowser(temporary_dict))
                
                temp_trans_pairs = {epoch: reconstruction_pairs[epoch]}
                temporary_dict = {'translation_pairs':temp_trans_pairs}
                
                for i in range(sentence_vectors.shape[0]):
                    print("target: " +str(' '.join([LFidx2vocab[sentence_vectors[i][j].item()] for j in range(sentence_vectors.shape[1]) if sentence_vectors[i][j].item() != 0])))
                    temp_list = []
                    for j in range(tensor_of_all_AUTOdecoded_outputs.topk(1)[1].squeeze(2).cpu().numpy().shape[1]):
                        temp_list.append(LFidx2vocab[tensor_of_all_AUTOdecoded_outputs.topk(1)[1].cpu().numpy().squeeze(2)[i][j]])
                    #print("predicted: " + str([LFidx2vocab[out_list[i].item()] for i in range(len(out_list))]))
                    print("predicted: " + str(' '.join(temp_list)))
                    print(' ')
                        
                print("Reconstruction ACCURACY FOR EPOCH ", epoch+1, " : ", acc_bowser(temporary_dict))


                validation_loss_dict['reconstruction'].append(board_reconstruction_loss.item()/len(validation_sampled))
                validation_loss_dict['translation'].append(board_translation_loss.item()/len(validation_sampled))
                validation_loss_dict['joint'].append(board_joint_loss.item()/len(validation_sampled))
                if kl_bool == 1:
                    validation_loss_dict['kl'].append(board_kl.item()/len(validation_sampled))

 

# =============================================================================
#                     if test_batch_num %20 == 0:
#                         temp_trans_pairs = {epoch: reconstruction_pairs[epoch]}
#                         temporary_dict = {'translation_pairs':temp_trans_pairs}
#                         
#                         for i in range(x.shape[0]):
#                             print("target: " +str(' '.join([LFidx2vocab[x[i][j].item()] for j in range(x.shape[1]) if x[i][j].item() != 0])))
#                             temp_list = []
#                             for j in range(predicted_auto.shape[1]):
#                                 temp_list.append(LFidx2vocab[predicted_auto[i][j]])
#                             #print("predicted: " + str([LFidx2vocab[out_list[i].item()] for i in range(len(out_list))]))
#                             print("predicted: " + str(' '.join(temp_list)))
#                             print(' ')
#                             
#                         print("VALIDATION ACCURACY FOR EPOCH ", epoch+1, " : ", acc_bowser(temporary_dict))
# 
# =============================================================================
                        
            if kl_bool ==1:
                validation_loss_dict['kl'].append(board_kl.item()/test_batch_num)
                    
                #test_loss_list.append(current_loss_list)
        if args.save_every_epoch ==1 and epoch>= args.save_from:
            save_every_epoch(translation_pairs, reconstruction_pairs, args, epoch)
        else:
            save(translation_pairs, reconstruction_pairs, args)
        log(args, epoch)
        
            
        
    
    #writer.close()
    

if __name__ == '__main__':
    print("running")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        USE_CUDA = True
    else:
        device = torch.device('cpu')
        USE_CUDA = False
    if args.cuda is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)

    num_epochs = 200
    train(num_epochs, args)
    
    
