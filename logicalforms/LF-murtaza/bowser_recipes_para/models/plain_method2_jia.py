import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import time


import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import time
from models.seq2seq_Luong import *


class CopyEncoderRAW(nn.Module):

    def __init__(self, hidden_dim, vocab_size, BERTWord=0, weight=None, lstm=False, word2vec=0, word_vectors=None):
        super(CopyEncoderRAW, self).__init__()
        self.hidden_dim = hidden_dim
        if BERTWord==1:
            self.BertLayer = nn.Linear(1024, hidden_dim)
        self.embedding =  nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        if BERTWord==1:
            self.embedding = self.embedding.from_pretrained(weight)
        self.BERTWord = BERTWord
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True,
                bidirectional=True)
        
        self.lstm = lstm
        self.word2vec = word2vec
        if word2vec == 1:
            word_vectors = torch.FloatTensor(word_vectors).cuda()
            self.embedding = self.embedding.from_pretrained(word_vectors)
            self.word2vec_lin = nn.Linear(300, hidden_dim)


       #sentence is list of index of words
    def forward(self, sentence2idxvec,X_lengths, hidden=None):
        #print("input shape:" + str(sentence2idxvec.shape))        
        embeds = self.embedding(sentence2idxvec)
        if self.BERTWord == 1:
            embeds = self.BertLayer(embeds)
        #embeds_return = embeds
        #print("embeds:" + str(embeds.shape))
        if self.word2vec == 1:
            embeds = self.word2vec_lin(embeds)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, X_lengths, batch_first=True)
        #real_input = embeds
        #packed = torch.nn.utils.rnn.pack_padded_sequence(real_input, X_lengths, batch_first=True)
        #print("packed shape:" + str(packed.shape))
        #print("real input shape:" + str(real_input.view(batch_size, len(sentence2idxvec), -1).shape))
        #print("real input view" + str(real_input.view(batch_size, len(sentence2idxvec), -1)))
        #lstm_out, hidden = self.gru(real_input.view(batch_size, len(sentence2idxvec), -1))#out = b(1) x seq x hid*2, hidden = b(1) x 1 x hid*2
        lstm_out, hidden = self.gru(embeds, hidden)
        #print("passed gru!")
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        #lstm_out= lstm_out[:, :, :self.hidden_dim] + lstm_out[:, :, self.hidden_dim:]  # Sum bidirectional outputs
        if self.lstm == 1:
            hidden = hidden[0]
        #Put in pad_packed 반대로!
        return lstm_out, hidden

class AutoDecoder(nn.Module):
    def __init__(self, AUTOhidden_dim, AUTOoutput_dim, bi =0):
        super(AutoDecoder, self).__init__()
        self.hidden_dim = AUTOhidden_dim
        self.embedding =  nn.Embedding(AUTOoutput_dim, AUTOhidden_dim, padding_idx=0)
        self.gru = nn.GRU(AUTOhidden_dim, AUTOhidden_dim, batch_first = True)
        self.out = nn.Linear(AUTOhidden_dim, AUTOoutput_dim)
        self.softmax = nn.LogSoftmax(dim=1) #Watch the dimension!

    def forward(self, input, hidden, encoder_outputs=None):
        batch_size = input.shape[0]
        hidden = hidden.view(1,batch_size, self.hidden_dim)
        output = self.embedding(input).view(batch_size, 1, -1)
        output = F.relu(output)
        output = output.contiguous(); hidden = hidden.contiguous()
        output, hidden = self.gru(output, hidden)
        output = output.contiguous()
        output = self.softmax(self.out(output.view(-1, output.shape[2]))) 
        kl = None
        return output, hidden, kl

class AutoDecoderAttn(AutoDecoder):
    def __init__(self, AUTOhidden_dim, AUTOoutput_dim):
        super(AutoDecoderAttn, self).__init__()
        AutoDecoder.__init__(self,AUTOhidden_dim, AUTOoutput_dim)
        
    def forward(self, input, hidden, encoder_outputs=None):
        
        batch_size = input.shape[0]
        hidden = hidden.view(1,batch_size, self.hidden_dim)
        output = self.embedding(input).view(batch_size, 1, -1)
        output = F.relu(output)
        output = output.contiguous(); hidden = hidden.contiguous()
        output, hidden = self.gru(output, hidden)
        output = output.contiguous()
        output = self.softmax(self.out(output.view(-1, output.shape[2]))) 
        kl = None
        return output, hidden, kl

class VAEDecoder(nn.Module):
    def vae_to_var(x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    
    def __init__(self, AUTOhidden_dim, AUTOoutput_dim, latent_dim, bi=0):
        super(VAEDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden2mean = nn.Linear(AUTOhidden_dim * 2, latent_dim)
        self.hidden2logv = nn.Linear(AUTOhidden_dim * 2, latent_dim)
        self.hidden_dim = AUTOhidden_dim
        self.latent2hidden = nn.Linear(latent_dim, AUTOhidden_dim)
        self.embedding =  nn.Embedding(AUTOoutput_dim, AUTOhidden_dim, padding_idx=0)
        self.gru = nn.GRU(AUTOhidden_dim, AUTOhidden_dim, batch_first = True)
        self.out = nn.Linear(AUTOhidden_dim, AUTOoutput_dim)
        self.softmax = nn.LogSoftmax(dim=1) #Watch the dimension!
        #TODO here
        #if bi ==1:
            

    def forward(self, input, hidden, order):
        if order == 0:
            hidden = hidden.view(batch_size, -1)
            mean = self.hidden2mean(hidden)
            logv = self.hidden2logv(hidden)
            std = torch.exp(0.5 * logv)
            
            z = torch.randn([batch_size, self.latent_dim]).cuda()
            z = z * std + mean
            hidden = self.latent2hidden(z)
        hidden = hidden.view(1,batch_size, self.hidden_dim)
        output = self.embedding(input).view(batch_size, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = output.contiguous()
        log_p_shape = output.shape[2]
        output = self.softmax(self.out(output.view(-1, output.shape[2]))) 
        global kl
        if order ==0:
            kl = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        return output, hidden, kl

class VAESeq2seqDecoder(nn.Module):
    def vae_to_var(x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    
    def __init__(self, AUTOhidden_dim, AUTOoutput_dim, latent_dim, bi=0):
        super(VAESeq2seqDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden2mean = nn.Linear(AUTOhidden_dim * 2, latent_dim)
        self.hidden2logv = nn.Linear(AUTOhidden_dim * 2, latent_dim)
        self.hidden_dim = AUTOhidden_dim
        self.latent2hidden = nn.Linear(latent_dim, AUTOhidden_dim)
        self.embedding =  nn.Embedding(AUTOoutput_dim, AUTOhidden_dim, padding_idx=0)
        self.gru = nn.GRU(AUTOhidden_dim, AUTOhidden_dim, batch_first = True)
        self.out = nn.Linear(AUTOhidden_dim, AUTOoutput_dim)
        self.softmax = nn.LogSoftmax(dim=1) #Watch the dimension!
        #TODO here
        #if bi ==1:
            

    def forward(self, input, hidden, order, batch_size):
        if order == 0:
            hidden = hidden.view(batch_size, -1)
            mean = self.hidden2mean(hidden)
            logv = self.hidden2logv(hidden)
            std = torch.exp(0.5 * logv)
            
            z = torch.randn([batch_size, self.latent_dim]).cuda()
            z = z * std + mean
            hidden = self.latent2hidden(z)
        hidden = hidden.view(1,batch_size, self.hidden_dim)
        output = self.embedding(input).view(batch_size, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = output.contiguous()
        log_p_shape = output.shape[2]
        #output = self.softmax(self.out(output.view(-1, output.shape[2]))) 
        output = self.out(output.view(-1, output.shape[2]))
        global kl
        if order ==0:
            kl = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        return output, hidden, kl





class CopyDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, max_oovs=12, bi =0, BERTWord=0, weight=None, lstm=False):
        super(CopyDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.time = time.time()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.BERTWord = BERTWord
        if BERTWord==1:
            self.embed = self.embed.from_pretrained(weight)
            self.Bert_layer = nn.Linear(1024, hidden_size)
        if lstm:
            self.gru = nn.LSTM(input_size=embed_size+hidden_size*2,
                hidden_size=hidden_size, batch_first=True)
        else:
            self.gru = nn.GRU(input_size=embed_size+hidden_size*2,
                hidden_size=hidden_size, batch_first=True)
        self.lstm = lstm
        self.max_oovs = max_oovs # largest number of OOVs available per sample

        # weights
        self.attn = Attn('general', hidden_size, bi=bi)
        
# =============================================================================
#         if bi == 1:
#                 self.Ws = nn.Linear(2*hidden_size, hidden_size)
#         else:
#                 self.Ws = nn.Linear(hidden_size, hidden_size) # only used at initial stage
#                 
# =============================================================================
        self.Ws = nn.Linear(hidden_size*2, hidden_size) # only used at initial stage
        
        self.Wo = nn.Linear(hidden_size*3, vocab_size) # generate mode
        self.Wc = nn.Linear(hidden_size*2, hidden_size*3) # copy mode
        self.nonlinear = nn.Tanh()
        
        #self.W_bi = nn.Linear(hidden_size*2, hidden_size)


    def forward(self, input_idx, encoded, encoded_idx, prev_state, weighted, order, X_lengths=None, train_mode = True):
        # input_idx(y_(t-1)): [b]			<- idx of next input to the decoder (Variable)
        # encoded: [b x seq x hidden]		<- hidden states created at encoder (Variable)
        # encoded_idx: [b x seq]			<- idx of inputs used at encoder (numpy)
        # prev_state(s_(t-1)): [1 x b x hidden]		<- hidden states to be used at decoder (Variable)
        # weighted: [b x 1 x hidden]		<- weighted attention of previous state, init with all zeros (Variable)

        #print("input_idx: " + str(input_idx))
        # hyperparameters
        start = time.time()
        time_check = False
        b = encoded.size(0) # batch size
        seq = encoded.size(1) # input sequence length
        vocab_size = self.vocab_size
        hidden_size = self.hidden_size

        # 0. set initial state s0 and initial attention (blank)
        
        
        if order==0:
            #print("prev state: ", prev_state)
            if self.lstm == False:
                prev_state = F.tanh(self.Ws(prev_state))
            else:
                prev_state = (F.tanh(self.Ws(prev_state)), torch.randn(self.Ws(prev_state).shape).unsqueeze(0).cuda())
         
        if self.lstm == False:
            prev_h = prev_state
        else:
            prev_h = prev_state[0]   
        

        prev_h = prev_h.unsqueeze(0) # [1 x b x hidden]
        if time_check:
            self.elapsed_time('state 0')

        # Jia: Context First 
        attn_weights = self.attn(prev_h, encoded) #attn_weights : [Batch x 1 x Seq]
        prev_h = prev_h.squeeze(0) 
        context = attn_weights.bmm(encoded)  #context: [Batch x 1 x Hidden]
        context = context.squeeze(1) 
        concat_input = torch.cat((prev_h, context), 1)
        
        # Finally predict next token (Luong eq. 6, without softmax)
        score_g = self.Wo(concat_input) #score_g should be is : [Batch*vocab_size] (Correct but before lstm)
        
        score_c = attn_weights.squeeze(1) #[Batch*seq]
        encoded_mask = torch.Tensor(np.array(encoded_idx==0, dtype=float)*(-1000)) # [b x seq]
        encoded_mask = self.to_cuda(encoded_mask)
        encoded_mask = Variable(encoded_mask)
        score_c = score_c + encoded_mask # padded parts will get close to 0 when applying softmax   
        
        score = torch.cat([score_g,score_c],1) # [b x (vocab+seq)]
        probs = F.softmax(score)
        prob_g = probs[:,:vocab_size] # [b x vocab]
        prob_c = probs[:,vocab_size:] # [b x seq]
        
        
        en = self.to_cuda(torch.LongTensor(encoded_idx)) # [b x in_seq]
        en.unsqueeze_(2) # [b x in_seq x 1]
        one_hot = self.to_cuda(torch.FloatTensor(en.size(0),en.size(1),prob_g.size(1))).zero_() # [b x in_seq x vocab]
        one_hot.scatter_(2,en,1) # one hot tensor: [b x seq x vocab]
        one_hot = self.to_cuda(one_hot)
        prob_c_to_g = torch.bmm(prob_c.unsqueeze(1),Variable(one_hot, requires_grad=False)) # [b x 1 x vocab]
        #prob_c_to_g = prob_c_to_g.squeeze() # [b x vocab]
        prob_c_to_g = prob_c_to_g.view(b,-1)

        out =  prob_g + prob_c_to_g
        #out = prob_g
        out = out.unsqueeze(1) # [b x 1 x vocab]

        #out is final probability 
        if train_mode == False:
            input_idx = out.squeeze(1).max(1)[1]
            #print('input_idx :', input_idx )
        
        #do lstm and output next state now 
        if self.BERTWord == 1:
            gru_input = self.Bert_layer(self.embed(input_idx)).view(b,1,-1)
        else:
            gru_input = self.embed(input_idx).view(b,1,-1) # [b x 1 x (h+emb)]
        #print("gru input: ", gru_input.shape)
        #print("context: ", context.view(b, 1, -1).shape)
        gru_input = torch.cat([gru_input, context.view(b, 1, -1)], 2)
        
        #print("prev_state shape:", prev_h.shape)
        prev_h = prev_h.unsqueeze(0)
        if self.lstm == False:
            prev_state = prev_h 
        else:
            prev_h = prev_h.contiguous()
            prev_state = (prev_h, prev_state[1].contiguous())
            gru_input = gru_input.contiguous()
        rnn_output, state = self.gru(gru_input, prev_state) #output and state are literally the same
        if self.lstm == False:
            state = state.view(b,-1) # [b x h]
        else:
            state = (state[0].view(b,-1), state[1])
        
        rnn_output = rnn_output.view(1,b,rnn_output.shape[2]) #rnn_output: [1 x Batch x Hidden]

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # rnn_output is now: [Batch x Hidden]
        #concat_output = torch.tanh(self.concat(concat_input))


        ###################################
        ####################################
        
        

        weighted = None
        return out, state, weighted

    def to_cuda(self, tensor):
        # turns to cuda
        if torch.cuda.is_available():
            return tensor.cuda()
        else:
            return tensor
        #return tensor
    def elapsed_time(self, state):
        elapsed = time.time()
        print("Time difference from %s: %1.4f"%(state,elapsed-self.time))
        self.time = elapsed
        return
