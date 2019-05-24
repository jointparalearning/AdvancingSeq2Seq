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

    def __init__(self, hidden_dim, vocab_size):
        super(CopyEncoderRAW, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding =  nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True,
            bidirectional=True)

       #sentence is list of index of words
    def forward(self, sentence2idxvec,X_lengths):
        #print("input shape:" + str(sentence2idxvec.shape))
        embeds = self.embedding(sentence2idxvec)
        #embeds_return = embeds
        #print("embeds:" + str(embeds.shape))
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, X_lengths, batch_first=True)
        #real_input = embeds
        #packed = torch.nn.utils.rnn.pack_padded_sequence(real_input, X_lengths, batch_first=True)
        #print("packed shape:" + str(packed.shape))
        #print("real input shape:" + str(real_input.view(batch_size, len(sentence2idxvec), -1).shape))
        #print("real input view" + str(real_input.view(batch_size, len(sentence2idxvec), -1)))
        #lstm_out, hidden = self.gru(real_input.view(batch_size, len(sentence2idxvec), -1))#out = b(1) x seq x hid*2, hidden = b(1) x 1 x hid*2
        lstm_out, hidden = self.gru(embeds)
        #print("passed gru!")
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        #Put in pad_packed 반대로!
        return lstm_out, hidden

class AutoDecoder(nn.Module):
    def __init__(self, AUTOhidden_dim, AUTOoutput_dim):
        super(AutoDecoder, self).__init__()
        self.hidden_dim = AUTOhidden_dim
        self.embedding =  nn.Embedding(AUTOoutput_dim, AUTOhidden_dim, padding_idx=0)
        self.gru = nn.GRU(AUTOhidden_dim, AUTOhidden_dim, batch_first = True)
        self.out = nn.Linear(AUTOhidden_dim, AUTOoutput_dim)
        self.softmax = nn.LogSoftmax(dim=1) #Watch the dimension!

    def forward(self, input, hidden, batch_size):
        hidden = hidden.view(1,batch_size, self.hidden_dim)
        output = self.embedding(input).view(batch_size, 1, -1)
        output = F.relu(output)
        output = output.contiguous(); hidden = hidden.contiguous()
        output, hidden = self.gru(output, hidden)
        output = output.contiguous()
        output = self.softmax(self.out(output.view(-1, output.shape[2]))) 
        kl = None
        return output, hidden, kl

# =============================================================================
# class AutoDecoderLuong(LuongAttnDecoderRNN):
#     def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1):
#         self.attn_model = 'concat'
#         super(AutoDecoderLuong, self).__init__()
#         
#     def forward(self, input_seq, last_hidden, encoder_outputs):
#         # Note: we run this one step at a time
#         # input_seq : [batch_size], something like torch.LongTensor([SOS_token] * small_batch_size)
#         # last_hidden: last elemnt of decoder_outputs [1 x batch_size x hidden]
#         # encoder_outputs: [batch_size x seq x hidden]
# 
#         # Get the embedding of the current input word (last output word)
#         batch_size = input_seq.size(0) 
#         embedded = self.embedding(input_seq)
#         embedded = self.embedding_dropout(embedded)
#         #No need for batch_irst = True in the gru because of the next line
#         embedded = embedded.view(1, batch_size, self.hidden_size)  # embedded: [1 x Batch x hidden]
# 
#         # Get current hidden state from input word and last hidden state
#         rnn_output, hidden = self.gru(embedded, last_hidden) #rnn output: [1 x Batch x Hidden], hidden: [1 x Batch x Hidden]
# 
#         # Calculate attention from current RNN state and all encoder outputs;
#         # apply to encoder outputs to get weighted average
#         attn_weights = self.attn(rnn_output, encoder_outputs) #attn_weights : [Batch x 1 x Seq]
#         context = attn_weights.bmm(encoder_outputs)  #context: [Batch x 1 x Hidden]
# 
#         # Attentional vector using the RNN hidden state and context vector
#         # concatenated together (Luong eq. 5)
#         rnn_output = rnn_output.squeeze(0)  # rnn_output is now: [Batch x Hidden]
#         context = context.squeeze(1)  # context is now: [Batch x Hidden]
#         concat_input = torch.cat((rnn_output, context), 1)
#         concat_output = torch.tanh(self.concat(concat_input))
# 
#         # Finally predict next token (Luong eq. 6, without softmax)
#         output = self.out(concat_output) #output is : [Batch*output_size] (output size is the number of all vocabs)
# 
#         # Return final output, hidden state, and attention weights (for visualization)
#         kl = None
#         return output, hidden, kl
#         
# 
# =============================================================================
        
class AutoDecoderLuong(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(AutoDecoderLuong, self).__init__()

        attn_model = 'concat'
        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if self.attn_model != 'none':
            self.attn = Attn(self.attn_model, hidden_size)

        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # input_seq : [batch_size], something like torch.LongTensor([SOS_token] * small_batch_size)
        # last_hidden: last elemnt of decoder_outputs [1 x batch_size x hidden]
        # encoder_outputs: [batch_size x seq x hidden]

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0) 
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        #No need for batch_irst = True in the gru because of the next line
        embedded = embedded.view(1, batch_size, self.hidden_size)  # embedded: [1 x Batch x hidden]

        # Get current hidden state from input word and last hidden state
        last_hidden = last_hidden.view(1, batch_size, self.hidden_size)
        last_hidden =last_hidden.contiguous()
        rnn_output, hidden = self.gru(embedded, last_hidden) #rnn output: [1 x Batch x Hidden], hidden: [1 x Batch x Hidden]

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs) #attn_weights : [Batch x 1 x Seq]
        context = attn_weights.bmm(encoder_outputs)  #context: [Batch x 1 x Hidden]

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # rnn_output is now: [Batch x Hidden]
        context = context.squeeze(1)  # context is now: [Batch x Hidden]
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output) #output is : [Batch*output_size] (output size is the number of all vocabs)
        kl=None
        output = output.contiguous()
        output = self.softmax(output)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, kl


    
    
class CopyDecoder(nn.Module):
    def __init__(self, which_attn_g, which_attn_c, generate_bahd, copy_bahd, vocab_size, embed_size, hidden_size, max_oovs=12, local_attn_cp=0, D=0, bi =0):
        super(CopyDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.time = time.time()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.max_oovs = max_oovs # largest number of OOVs available per sample

        
        #self.Ws = nn.Linear(hidden_size, hidden_size) # only used at initial stage
        #self.Wo = nn.Linear(hidden_size*2, vocab_size) # generate mode
        #self.Wc = nn.Linear(hidden_size, hidden_size*2) # copy mode
        self.nonlinear = nn.Tanh()
        self.which_attn_g,self.which_attn_c  = which_attn_g, which_attn_c
        self.generate_bahd, self.copy_bahd = generate_bahd,copy_bahd
        # weights
        #self.attn = Attn('concat', hidden_size)
        self.attn = Attn(which_attn_g, hidden_size, bi=1)
        self.local_attn_cp = local_attn_cp
        if self.local_attn_cp != 0:
            self.sig = nn.Sigmoid()
            self.D = D
            self.Vp = nn.Parameter(torch.randn(self.hidden_size, 1))
            self.Wp = nn.Linear(self.hidden_size, self.hidden_size)

        #elif which_attn_g == 'dot':
        #    self.Wo = nn.Linear(hidden_size, vocab_size)
        
        if generate_bahd in [0,2]:
            self.gru = nn.GRU(input_size=embed_size+3*hidden_size, hidden_size=hidden_size, batch_first=True)
        #elif generate_bahd or copy_bahd:
        #    self.gru = nn.GRU(input_size=embed_size+hidden_size, hidden_size=hidden_size, batch_first=True)
        #else:
        #    self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
        elif generate_bahd ==1:
            self.gru = nn.GRU(input_size=embed_size+2*hidden_size, hidden_size=hidden_size, batch_first=True)
            
        if generate_bahd == 0:
            self.Wo = nn.Linear(hidden_size, vocab_size)
        elif generate_bahd in [1,2]:
            self.Wo = nn.Linear(hidden_size*3, vocab_size)
            
        if copy_bahd:
            self.Wc = nn.Linear(hidden_size*2, hidden_size)
            if which_attn_c == 'concat':
                self.Wc = nn.Linear(hidden_size*3, hidden_size)
                self.v = nn.Parameter(torch.randn(1,hidden_size,1))
            elif which_attn_c == 'location':
                self.Wc = nn.Linear(hidden_size, 1)

        else:
            self.Wc = nn.Linear(hidden_size, hidden_size*3)
            if which_attn_c == 'concat':
                self.Wc = nn.Linear(hidden_size*3, hidden_size*3)
                self.v = nn.Parameter(torch.randn(1,hidden_size*3,1))
            elif which_attn_c == 'location':
                self.Wc = nn.Linear(hidden_size, 1)
                
# =============================================================================
#         if bi == 1:
#             self.Ws = nn.Linear(2*hidden_size, hidden_size)
#         else:
# =============================================================================
        self.Ws = nn.Linear(2*hidden_size, hidden_size) # only used at initial stage
        
            
    def forward(self, input_idx, encoded, encoded_idx, prev_state, weighted, order, X_lengths, train_mode):
        # input_idx(y_(t-1)): [b]			<- idx of next input to the decoder (Variable)
        # encoded: [b x seq x hidden]		<- hidden states created at encoder (Variable)
        # encoded_idx: [b x seq]			<- idx of inputs used at encoder (numpy)
        # prev_state(s_(t-1)): [b x hidden]		<- hidden states to be used at decoder (Variable)
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
            prev_state = self.Ws(prev_state)
            weighted = torch.Tensor(b,1,hidden_size*2).zero_()
            weighted = self.to_cuda(weighted)
            weighted = Variable(weighted)


        prev_state = prev_state.unsqueeze(0) # [1 x b x hidden]
        if time_check:
            self.elapsed_time('state 0')

        # 1. update states
        
        gru_input = self.embed(input_idx).view(b,1,-1)
        if self.generate_bahd in [0,2]:
            attn_weights = self.attn(prev_state, encoded) #attn_weights : [Batch x 1 x Seq]
            context = attn_weights.bmm(encoded)  #context: [Batch x 1 x Hidden]

            gru_input = torch.cat([gru_input, context], 2) 
        
        #if self.copy_bahd:
        gru_input = torch.cat([gru_input, weighted], 2) 
        gru_input = gru_input.contiguous()
        prev_state = prev_state.contiguous()
        rnn_output, state = self.gru(gru_input, prev_state) #output and state are literally the same
        state = state.view(b,-1) # [b x h]
        rnn_output = rnn_output.view(1,b,rnn_output.shape[2]) #rnn_output: [1 x Batch x Hidden]

        #score_g is Luong 
        if self.generate_bahd in [1,2] :
            # 2. predict next word y_t
            # 2-1) get scores score_g for generation- mode
            #score_g = self.Wo(state) # [b x vocab_size]
            #Basically just do all of LuongAttnDecoderRnn
            #print("encoded: ", encoded)
            attn_weights = self.attn(rnn_output, encoded) #attn_weights : [Batch x 1 x Seq]
            context = attn_weights.bmm(encoded)  #context: [Batch x 1 x Hidden]
    
            # Attentional vector using the RNN hidden state and context vector
            # concatenated together (Luong eq. 5)
            rnn_output = rnn_output.squeeze(0)  # rnn_output is now: [Batch x Hidden]
            context = context.squeeze(1)  # context is now: [Batch x Hidden]
            concat_input = torch.cat((rnn_output, context), 1)
    
            # Finally predict next token (Luong eq. 6, without softmax)
            #This is just output, Luong step 4. 
            score_g = self.Wo(concat_input) #score_g should be is : [Batch*vocab_size] 
        
        elif self.generate_bahd ==0:
            rnn_output = rnn_output.squeeze(0)
            score_g = self.Wo(rnn_output)
        
        #print("score g that just came out: ", score_g)
        if time_check:
            self.elapsed_time('state 2-1')


        if self.copy_bahd:
            if self.which_attn_c == 'general':
                # 2-2) get scores score_c for copy mode, remove possibility of giving attention to padded values
                score_c = F.tanh(self.Wc(encoded.contiguous().view(-1,hidden_size*2))) # [b*seq x hidden_size]
                score_c = score_c.view(b,-1,hidden_size) # [b x seq x hidden_size]
                score_c = torch.bmm(score_c, state.unsqueeze(2)).view(b,-1) # [b x seq]
            elif self.which_attn_c == 'dot':
                score_c = F.tanh(encoded.contiguous().view(-1,hidden_size*2))
                score_c = score_c.view(b,-1,hidden_size) # [b x seq x hidden_size]
                score_c = torch.bmm(score_c, state.unsqueeze(2)).view(b,-1) # [b x seq]
            elif self.which_attn_c == 'concat':
                concatted = torch.cat((rnn_output.unsqueeze(1).expand(encoded.shape[0],encoded.shape[1],hidden_size), encoded.contiguous()), 2)
                score_c = F.tanh(self.Wc(concatted)) # score_c: [b x seq x hidden]
                score_c = score_c.bmm(self.v.expand(b,score_c.shape[2],1)).squeeze(2)
            elif self.which_attn_c == 'location':
                #torch.matmul(torch.eye(self.hidden_size), self.Wc)
                score_c = self.Wc(rnn_output.unsqueeze(1)).squeeze(2) #[b x seq]
                #kind of 애매  but just leave it 
                score_c = F.tanh(score_c)
                
            
        else:
            if self.which_attn_c == 'general':
                score_c = F.tanh(self.Wc(encoded.contiguous().view(-1,hidden_size*2))) # [b*seq x hidden_size]
                score_c = score_c.view(b,-1,hidden_size*3) # [b x seq x hidden_size*2]
                score_c = torch.bmm(score_c, torch.cat([state, context], dim=1).unsqueeze(2) ).view(b,-1) # score_c: [b x seq], state.unsqueeze(2): [b x h x 1]
            elif self.which_attn_c == 'dot':
                score_c = F.tanh(encoded.contiguous().view(-1,hidden_size*2))
                score_c = score_c.view(b,-1,hidden_size*3) # [b x seq x hidden_size*2]
                score_c = torch.bmm(score_c, torch.cat([state, context], dim=1).unsqueeze(2) ).view(b,-1) # score_c: [b x seq], state.unsqueeze(2): [b x h x 1]
            elif self.which_attn_c == 'concat':
                raise NotImplementedError
            elif self.which_attn_c == 'location':
                raise NotImplementedError

        if self.local_attn_cp ==1 :
            pt = F.tanh(self.Wp(state)).unsqueeze(1) #[Batch x 1 x hidden]
            pt = pt.bmm(self.Vp.expand(pt.shape[0], self.Vp.shape[0], self.Vp.shape[1])).squeeze(1) #[Batch x1]
            pt =  torch.tensor(X_lengths).float().cuda() * self.sig(pt).squeeze(1) #pt: [Batch]
            #print(pt)
            s = torch.tensor([[j >= pt[i].item() - self.D and j <= pt[i].item() + self.D for j in range(X_lengths[i].item())] + [0.0]*(encoded.shape[1]- X_lengths[i].item()) for i in range(encoded.shape[0])]).cuda() #[batch x length of each sequence]
            #print(s[1], X_lengths[1])
            #print(s[2], X_lengths[2])
            pt = pt.unsqueeze(1).expand(pt.shape[0],encoded.shape[1])
            pointer_coeffs = torch.exp(( s - pt )**2/(-2*(self.D/2)**2))
            score_c = score_c * pointer_coeffs

     

        #score_c = F.tanh(score_c) # purely optional....
        encoded_mask = torch.Tensor(np.array(encoded_idx==0, dtype=float)*(-1000)) # [b x seq]
        encoded_mask = self.to_cuda(encoded_mask)
        #print("encoded mask is " + str( encoded_mask))
        encoded_mask = Variable(encoded_mask)
        score_c = score_c + encoded_mask # padded parts will get close to 0 when applying softmax

        if time_check:
            self.elapsed_time('state 2-2')

        # 2-3) get softmax-ed probabilities
        score = torch.cat([score_g,score_c],1) # [b x (vocab+seq)]
        #print("score g is ", score_g)
        #print("score c is ", score_c)
        probs = F.softmax(score)
        prob_g = probs[:,:vocab_size] # [b x vocab]
        prob_c = probs[:,vocab_size:] # [b x seq]
        prob_c_given = prob_c / torch.sum(prob_c)

        if time_check:
            self.elapsed_time('state 2-3')

        # 2-4) add empty sizes to prob_g which correspond to the probability of obtaining OOV words
        oovs = Variable(torch.Tensor(b,self.max_oovs).zero_())+1e-4
        oovs = self.to_cuda(oovs)
        prob_g = torch.cat([prob_g,oovs],1)

        if time_check:
            self.elapsed_time('state 2-4')

        # 2-5) add prob_c to prob_g
        # prob_c_to_g = self.to_cuda(torch.Tensor(prob_g.size()).zero_())
        # prob_c_to_g = Variable(prob_c_to_g)
        # for b_idx in range(b): # for each sequence in batch
        # 	for s_idx in range(seq):
        # 		prob_c_to_g[b_idx,encoded_idx[b_idx,s_idx]]=prob_c_to_g[b_idx,encoded_idx[b_idx,s_idx]]+prob_c[b_idx,s_idx]


        # prob_c_to_g = Variable
        en = self.to_cuda(torch.LongTensor(encoded_idx)) # [b x in_seq]
        en.unsqueeze_(2) # [b x in_seq x 1]
        one_hot = self.to_cuda(torch.FloatTensor(en.size(0),en.size(1),prob_g.size(1))).zero_() # [b x in_seq x vocab]
        one_hot.scatter_(2,en,1) # one hot tensor: [b x seq x vocab]
        one_hot = self.to_cuda(one_hot)
        prob_c_to_g = torch.bmm(prob_c.unsqueeze(1),Variable(one_hot, requires_grad=False)) # [b x 1 x vocab]
        #prob_c_to_g = prob_c_to_g.squeeze() # [b x vocab]
        prob_c_to_g = prob_c_to_g.view(b,-1)
        
        out =  prob_g + prob_c_to_g
        #out = prob_c_to_g
        out = out.unsqueeze(1) # [b x 1 x vocab]

        if time_check:
            self.elapsed_time('state 2-5')

        # 3. get weighted attention to use for predicting next word
        # 3-1) get tensor that shows whether each decoder input has previously appeared in the encoder
        idx_from_input = []
        for i,j in enumerate(encoded_idx):
            idx_from_input.append([int(k==input_idx[i].item()) for k in j])
        idx_from_input = torch.Tensor(np.array(idx_from_input, dtype=float))
        # idx_from_input : np.array of [b x seq]
        idx_from_input = self.to_cuda(idx_from_input)
        idx_from_input = Variable(idx_from_input)
        #print(idx_from_input)
        for i in range(b):
            if idx_from_input[i].sum().item()>1:
                idx_from_input[i] = idx_from_input[i]/idx_from_input[i].sum().item()

        if time_check:
            self.elapsed_time('state 3-1')

        # 3-2) multiply with prob_c to get final weighted representation
        #attn = prob_c * idx_from_input
        attn = prob_c_given * idx_from_input
            
        # for i in range(b):
        # 	tmp_sum = attn[i].sum()
        # 	if (tmp_sum.data[0]>1e-6):
        # 		attn[i] = attn[i] / tmp_sum.data[0]
        attn = attn.unsqueeze(1) # [b x 1 x seq]
        weighted = torch.bmm(attn, encoded) # weighted: [b x 1 x hidden*2]

        if time_check:
            self.elapsed_time('state 3-2')

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
