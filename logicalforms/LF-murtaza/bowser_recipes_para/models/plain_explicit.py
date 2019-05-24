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


class CopyEncoderRAW(nn.Module):

    def __init__(self, hidden_dim, vocab_size):
        super(CopyEncoderRAW, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding =  nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim+100, hidden_dim, batch_first=True,
            bidirectional=True)

       #sentence is list of index of words
    def forward(self, sentence2idxvec, X_lengths):
        #print("input shape:" + str(sentence2idxvec.shape))
        embeds = self.embedding(sentence2idxvec)
        #embeds_return = embeds
        #print("embeds:" + str(embeds.shape))
        embeds = torch.cat([embeds, binary_vectors.view(binary_vectors.shape[0], binary_vectors.shape[1], -1).expand(binary_vectors.shape[0], binary_vectors.shape[1], 100)], dim=2)
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
        self.gru = nn.GRU(AUTOhidden_dim, AUTOhidden_dim*2, batch_first = True)
        self.out = nn.Linear(AUTOhidden_dim*2, AUTOoutput_dim)
        self.softmax = nn.LogSoftmax(dim=1) #Watch the dimension!

    def forward(self, input, hidden, X_lengths):
        hidden = hidden.view(1,batch_size, self.hidden_dim*2)
        output = self.embedding(input).view(batch_size, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = output.contiguous()
        output = self.softmax(self.out(output.view(-1, output.shape[2]))) 
        kl = None
        return output, hidden, kl


class CopyDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, max_oovs=12):
        super(CopyDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.time = time.time()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(input_size=embed_size+hidden_size*2+100,
            hidden_size=hidden_size, batch_first=True)
        self.max_oovs = max_oovs # largest number of OOVs available per sample

        # weights
        self.Ws = nn.Linear(hidden_size*2+100, hidden_size) # only used at initial stage
        self.Wo = nn.Linear(hidden_size, vocab_size) # generate mode
        self.Wc = nn.Linear(hidden_size*2+100, hidden_size) # copy mode
        self.nonlinear = nn.Tanh()

    def forward(self, input_idx, encoded, encoded_idx, prev_state, weighted, order, train_mode=None):
        # input_idx(y_(t-1)): [b]			<- idx of next input to the decoder (Variable)
        # encoded: [b x seq x hidden*2]		<- hidden states created at encoder (Variable)
        # encoded_idx: [b x seq]			<- idx of inputs used at encoder (numpy)
        # prev_state(s_(t-1)): [1 x b x hidden]		<- hidden states to be used at decoder (Variable)
        # weighted: [b x 1 x hidden*2]		<- weighted attention of previous state, init with all zeros (Variable)

        #print("input_idx: " + str(input_idx))
        # hyperparameters
        start = time.time()
        time_check = False
        b = encoded.size(0) # batch size
        seq = encoded.size(1) # input sequence length
        vocab_size = self.vocab_size
        hidden_size = self.hidden_size

        encoded = torch.cat([encoded, binary_vectors.view(binary_vectors.shape[0], binary_vectors.shape[1], -1).expand(binary_vectors.shape[0], binary_vectors.shape[1], 100)], dim=2)
        # 0. set initial state s0 and initial attention (blank)
        if order==0:
            prev_state = self.Ws(encoded[:,-1])
            weighted = torch.Tensor(b,1,hidden_size*2+100).zero_()
            weighted = self.to_cuda(weighted)
            weighted = Variable(weighted)

        prev_state = prev_state.unsqueeze(0) # [1 x b x hidden]
        if time_check:
            self.elapsed_time('state 0')

        # 1. update states
        gru_input = torch.cat([self.embed(input_idx).view(b,1,-1), weighted],2) # [b x 1 x (h*2+emb+100)]
        
        _, state = self.gru(gru_input, prev_state)
        state = state.view(b,-1) # [b x h]

        if time_check:
            self.elapsed_time('state 1')

        # 2. predict next word y_t
        # 2-1) get scores score_g for generation- mode
        score_g = self.Wo(state) # [b x vocab_size]

        if time_check:
            self.elapsed_time('state 2-1')

        # 2-2) get scores score_c for copy mode, remove possibility of giving attention to padded values
        score_c = F.tanh(self.Wc(encoded.contiguous().view(-1,hidden_size*2+100))) # [b*seq x hidden_size]
        score_c = score_c.view(b,-1,hidden_size) # [b x seq x hidden_size]
        score_c = torch.bmm(score_c, state.unsqueeze(2)).view(b,-1) # [b x seq]

        score_c = F.tanh(score_c) # purely optional....

        encoded_mask = torch.Tensor(np.array(encoded_idx==0, dtype=float)*(-1000)) # [b x seq]
        encoded_mask = self.to_cuda(encoded_mask)
        #print("encoded mask is " + str( encoded_mask))
        encoded_mask = Variable(encoded_mask)
        score_c = score_c + encoded_mask # padded parts will get close to 0 when applying softmax

        if time_check:
            self.elapsed_time('state 2-2')

        # 2-3) get softmax-ed probabilities
        score = torch.cat([score_g,score_c],1) # [b x (vocab+seq)]
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
        attn = prob_c_given * idx_from_input
        # for i in range(b):
        # 	tmp_sum = attn[i].sum()
        # 	if (tmp_sum.data[0]>1e-6):
        # 		attn[i] = attn[i] / tmp_sum.data[0]
        attn = attn.unsqueeze(1) # [b x 1 x seq]
        weighted = torch.bmm(attn, encoded) # weighted: [b x 1 x hidden*2+1]

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
