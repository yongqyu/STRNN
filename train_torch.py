#! /usr/bin/env python

import os
import datetime
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import data_loader

# Parameters
# ==================================================
ftype = torch.cuda.FloatTensor
ltype = torch.cuda.LongTensor

# Data loading params
train_file = "./prepro_train.txt"
valid_file = "./prepro_valid.txt"
test_file = "./prepro_test.txt"

# Model Hyperparameters
dim = 13    # dimensionality
ww = 360  # winodw width (6h)
up_time = 560632.0  # min
lw_time = 0.
up_dist = 457.335   # km
lw_dist = 0.
reg_lambda = 0.0005

# Training Parameters
batch_size = 2
num_epochs = 30
learning_rate = 0.001
momentum = 0.9
evaluate_every = 1
h_0 = Variable(torch.randn(dim, 1), requires_grad=False).type(ftype)

user_cnt = 107092
loc_cnt = 1280969

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
train_user, train_td, train_ld, train_loc = data_loader.treat_prepro(train_file)
valid_user, valid_td, valid_ld, valid_loc = data_loader.treat_prepro(valid_file)
test_user, test_td, test_ld, test_loc = data_loader.treat_prepro(test_file)

print("User/Location: {:d}/{:d}/{:d}".format(user_cnt, loc_cnt, len(train_user)))
print("==================================================================================")

class STRNNCell(nn.Module):
    def __init__(self, hidden_size):
        super(STRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # C
        self.weight_th_upper = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # T
        self.weight_th_lower = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # T
        self.weight_sh_upper = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # S
        self.weight_sh_lower = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # S

        self.location_weight = nn.Embedding(loc_cnt, hidden_size)
        self.permanet_weight = nn.Embedding(user_cnt, hidden_size)

        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, td_upper, td_lower, ld_upper, ld_lower, loc, hx):
        loc_len = len(loc)
        Ttd = [((self.weight_th_upper*td_upper[i] + self.weight_th_lower*td_lower[i])\
                /(td_upper[i]+td_lower[i])) for i in xrange(loc_len)]
        Sld = [((self.weight_sh_upper*ld_upper[i] + self.weight_sh_lower*ld_lower[i])\
                /(ld_upper[i]+ld_lower[i])) for i in xrange(loc_len)]
        
        loc = self.location_weight(loc).view(-1,self.hidden_size,1)
        loc_vec = torch.sum(torch.cat([torch.mm(Sld[i], torch.mm(Ttd[i], loc[i]))\
                .view(1,self.hidden_size,1) for i in xrange(loc_len)], dim=0), dim=0)
        usr_vec = torch.mm(self.weight_ih, hx) 
        hx = loc_vec + usr_vec # hidden_size x 1
        return self.sigmoid(hx)

    def loss(self, user, td_upper, td_lower, ld_upper, ld_lower, loc, hx):
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        p_u = self.permanet_weight(user)
        q_v = self.location_weight(loc)
        output = torch.mm(q_v, (h_tq + torch.t(p_u)))

        l2_reg = None
        for W in self.parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg += W.norm(2)

        return torch.log(1+torch.exp(torch.neg(output))) + l2_reg.pow(2) * reg_lambda

    def validation(self, user, hx):
        p_u = self.permanet_weight(user)
        user_vector = hx + torch.t(p_u)
        ret = torch.sum(torch.mul(user_vector.view(-1), self.location_weight.weight)\
                , dim=1, keepdim=True).data.cpu().numpy()
        return np.argsort(np.squeeze(-1*ret))

###############################################################################################
def parameters():
    params = []
    for model in [strnn_model]:
        params += list(model.parameters())

    return params

def print_score(batches, step):
    recall1 = 0.
    recall5 = 0.
    recall10 = 0.
    recall100 = 0.
    recall1000 = 0.
    recall10000 = 0.

    for i, batch in enumerate(batches):
        batch_user, batch_td, batch_ld, batch_loc = batch
        batch_o, target = run(batch_user, batch_td, batch_ld, batch_loc, step=step)

        recall1 += target in batch_o[:1]
        recall5 += target in batch_o[:5]
        recall10 += target in batch_o[:10]
        recall100 += target in batch_o[:100]
        recall1000 += target in batch_o[:1000]
        recall10000 += target in batch_o[:10000]

    print("recall@1: ", recall1/i)
    print("recall@5: ", recall5/i)
    print("recall@10: ", recall10/i)
    print("recall@100: ", recall100/i)
    print("recall@1000: ", recall1000/i)
    print("recall@10000: ", recall10000/i)

###############################################################################################
def run(user, td, ld, loc, step):

    optimizer.zero_grad()
    
    seqlen = len(td)
    user = Variable(torch.from_numpy(np.asarray([user]))).type(ltype)

    #neg_loc = Variable(torch.FloatTensor(1).uniform_(0, len(poi2pos)-1).long()).type(ltype)
    #(neg_lati, neg_longi) = poi2pos.get(neg_loc.data.cpu().numpy()[0])
    rnn_output = h_0
    for idx in xrange(seqlen-1):
        td_upper = Variable(torch.from_numpy(np.asarray(up_time-td[idx]))).type(ftype)
        td_lower = Variable(torch.from_numpy(np.asarray(td[idx]-lw_time))).type(ftype)
        ld_upper = Variable(torch.from_numpy(np.asarray(up_dist-ld[idx]))).type(ftype)
        ld_lower = Variable(torch.from_numpy(np.asarray(ld[idx]-lw_dist))).type(ftype)
        location = Variable(torch.from_numpy(np.asarray(loc[idx]))).type(ltype)
        rnn_output = strnn_model(td_upper, td_lower, ld_upper, ld_lower, location, rnn_output)#, neg_lati, neg_longi, neg_loc, step)

    if step > 1:
        return strnn_model.validation(user, rnn_output), loc[-1]

    td_upper = Variable(torch.from_numpy(np.asarray(up_time-td[-1]))).type(ftype)
    td_lower = Variable(torch.from_numpy(np.asarray(td[-1]-lw_time))).type(ftype)
    ld_upper = Variable(torch.from_numpy(np.asarray(up_dist-ld[-1]))).type(ftype)
    ld_lower = Variable(torch.from_numpy(np.asarray(ld[-1]-lw_dist))).type(ftype)
    location = Variable(torch.from_numpy(np.asarray(loc[-1]))).type(ltype)
    J = strnn_model.loss(user, td_upper, td_lower, ld_upper, ld_lower, location[-1], rnn_output)#, neg_lati, neg_longi, neg_loc, step)
    
    J.backward()
    optimizer.step()
    
    return J.data.cpu().numpy()

###############################################################################################
strnn_model = STRNNCell(dim).cuda()
optimizer = torch.optim.SGD(parameters(), lr=learning_rate, momentum=momentum)

for i in xrange(num_epochs):
    # Training
    total_loss = 0.
    train_batches = list(zip(train_user, train_td, train_ld, train_loc))
    for j, train_batch in enumerate(train_batches):
        #inner_batches = data_loader.inner_iter(train_batch, batch_size)
        #for k, inner_batch in inner_batches:
        batch_user, batch_td, batch_ld, batch_loc = train_batch#inner_batch)
        total_loss += run(batch_user, batch_td, batch_ld, batch_loc, step=1)
        if (j+1) % 200 == 0:
            print("batch #{:d}: ".format(j+1)), "batch_loss :", total_loss/j, datetime.datetime.now()
    # Evaluation
    if (i+1) % evaluate_every == 0:
        print("==================================================================================")
        print("Evaluation at epoch #{:d}: ".format(i+1)), total_loss/j, datetime.datetime.now()
        valid_batches = list(zip(valid_user, valid_td, valid_ld, valid_loc))
        print_score(valid_batches, step=2)

# Testing
print("Training End..")
print("==================================================================================")
print("Test: ")
test_batches = list(zip(test_user, test_td, test_ld, test_loc))
print_score(test_batches, step=3)