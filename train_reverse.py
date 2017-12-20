#! /usr/bin/env python

import os
import datetime
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
train_file = "../dataset/loc-gowalla_totalCheckins.txt"

# Model Hyperparameters
dim = 13    # dimensionality
ww = 360  # winodw width (6h)
up_time = 1440  # 1d
lw_time = 30    # 30m
up_dist = 100   # ??
lw_dist = 1

# Training Parameters
batch_size = 2
num_epochs = 30
learning_rate = 0.001
momentum = 0.9
evaluate_every = 1

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
user_cnt, poi2pos, train_user, train_time, train_lati, train_longi, train_loc, valid_user, valid_time, valid_lati, valid_longi, valid_loc, test_user, test_time, test_lati, test_longi, test_loc = data_loader.load_data(train_file)

print("User/Location: {:d}/{:d}".format(user_cnt, len(poi2pos)))
print("==================================================================================")

class STRNNModule(nn.Module):
    def __init__(self):
        super(STRNNModule, self).__init__()

        # embedding:
        self.user_weight = Variable(torch.randn(user_cnt, dim), requires_grad=False).type(ftype)
        self.h_0 = Variable(torch.randn(dim, 1), requires_grad=False).type(ftype)
        self.location_weight = nn.Embedding(len(poi2pos), dim)
        self.perm_weight = nn.Embedding(user_cnt, dim)
        # attributes:
        self.time_upper = nn.Parameter(torch.randn(dim, dim).type(ftype))
        self.time_lower = nn.Parameter(torch.randn(dim, dim).type(ftype))
        self.dist_upper = nn.Parameter(torch.randn(dim, dim).type(ftype))
        self.dist_lower = nn.Parameter(torch.randn(dim, dim).type(ftype))
        self.C = nn.Parameter(torch.randn(dim, dim).type(ftype))

        # modules:
        self.sigmoid = nn.Sigmoid()

    # find the most closest value to w, w_cap(index)
    def find_w_cap(self, times, i):
        trg_t = times[i] - ww
        tmp_t = times[i]
        tmp_i = i-1
        for idx, t_w in enumerate(reversed(times[:i]), start=1):
            if t_w.data.cpu().numpy() == trg_t.data.cpu().numpy():
                return i-idx
            elif t_w.data.cpu().numpy() > trg_t.data.cpu().numpy():
                tmp_t = t_w
                tmp_i = i-idx
            elif t_w.data.cpu().numpy() < trg_t.data.cpu().numpy():
                if trg_t.data.cpu().numpy() - t_w.data.cpu().numpy() \
                    < tmp_t.data.cpu().numpy() - trg_t.data.cpu().numpy():
                    return i-idx 
                else:
                    return tmp_i 
        return 0

    def return_h_tw(self, times, latis, longis, locs, idx):
        w_cap = self.find_w_cap(times, idx)
        if w_cap is 0:
            return self.h_0
        else:
            h__tw = self.return_h_tw(times, latis, longis, locs, w_cap)
        user_vector = torch.mm(self.C, h__tw) # 13*13 x 13*1

        lati = latis[idx] - latis[w_cap:idx]
        longi = longis[idx] - longis[w_cap:idx]
        td = times[idx] - times[w_cap:idx]
        loc_vector = self.get_location_vector(td, self.euclidean_dist(lati, longi), locs[w_cap:idx])
        
        return self.sigmoid(loc_vector+user_vector)

    # get transition matrices by linear interpolation
    def get_location_vector(self, td, ld, locs):
        tud = up_time - td
        tdd = td - lw_time
        lud = up_dist - ld
        ldd = ld - lw_dist
        loc_vec = 0
        for i in xrange(len(tud)):
            Tt = torch.div(torch.mul(self.time_upper, tud[i]) + torch.mul(self.time_lower, tdd[i]),
                            tud[i]+tdd[i])
            Sl = torch.div(torch.mul(self.dist_upper, lud[i]) + torch.mul(self.dist_lower, ldd[i]),
                            lud[i]+ldd[i])
            loc_vec += torch.mm(Sl, torch.mm(Tt, torch.t(self.location_weight(locs[i]))))
        return loc_vec

    def euclidean_dist(self, x, y):
        return torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))

    def forward(self, user, times, latis, longis, locs, step):#neg_lati, neg_longi, neg_loc, step):
        # positive sampling
        pos_h = self.return_h_tw(times, latis, longis, locs, len(times)-1)

        '''
        # negative sampling
        w = self.find_w_cap(times.data.cpu().numpy(), times[-1].data.cpu().numpy(), len(times)-1)
        user_vector = torch.mm(self.C, user_vectors[w])

        lati = neg_lati - latis[w:-1]
        longi = neg_longi - longis[w:-1]
        td = times[-1] - times[w:-1]
        loc_vector = self.get_location_vector(td, self.euclidean_dist(lati, longi), locs[w:-1])
        neg_h = self.sigmoid(loc_vector+user_vector)
        '''

        # final prediction
        permanent = torch.t(self.perm_weight(user))
        if step is 1:
            output = torch.mm(self.location_weight(locs[-1]), pos_h + permanent)
            #output -= torch.mm(self.location_weight(neg_loc), neg_h + permanent)
            return output
        else:
            output = torch.mm(self.location_weight.weight, pos_h + permanent)
            #output -= torch.mm(self.location_weight(neg_loc), neg_h + permanent)
            return output

    # for test, process w/o last location
    def validation(self, user_vector):
        return torch.sum(torch.mul(user_vector, self.location_weight.weight), dim=1, keepdim=True)

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

    if step == 2:
        batch_user = valid_user
    elif step == 3:
        batch_user = test_user
        
    for i, batch in enumerate(batches):
        batch_time, batch_lati, batch_longi, batch_loc = batch
        batch_o, target = run(batch_user[i], batch_time, batch_lati, batch_longi, batch_loc, step=step)

        recall1 += target in np.argsort(np.squeeze(-1*batch_o))[:1]
        recall5 += target in np.argsort(np.squeeze(-1*batch_o))[:5]
        recall10 += target in np.argsort(np.squeeze(-1*batch_o))[:10]
        recall100 += target in np.argsort(np.squeeze(-1*batch_o))[:100]
        recall1000 += target in np.argsort(np.squeeze(-1*batch_o))[:1000]
        recall10000 += target in np.argsort(np.squeeze(-1*batch_o))[:10000]

    print("recall@1: ", recall1/i)
    print("recall@5: ", recall5/i)
    print("recall@10: ", recall10/i)
    print("recall@100: ", recall100/i)
    print("recall@1000: ", recall1000/i)
    print("recall@10000: ", recall10000/i)

###############################################################################################
def run(user, time, lati, longi, loc, step):

    optimizer.zero_grad()

    user = Variable(torch.from_numpy(np.asarray([user]))).type(ltype)
    time = Variable(torch.from_numpy(np.asarray(time))).type(ftype)
    lati = Variable(torch.from_numpy(np.asarray(lati))).type(ftype)
    longi = Variable(torch.from_numpy(np.asarray(longi))).type(ftype)
    loc = Variable(torch.from_numpy(np.asarray(loc))).type(ltype)

    #neg_loc = Variable(torch.FloatTensor(1).uniform_(0, len(poi2pos)-1).long()).type(ltype)
    #(neg_lati, neg_longi) = poi2pos.get(neg_loc.data.cpu().numpy()[0])
    rnn_output = strnn_model(user, time, lati, longi, loc, step)#, neg_lati, neg_longi, neg_loc, step)

    if step > 1:
        return rnn_output.data.cpu().numpy(), loc[-1].data.cpu().numpy()

    # Need to regularization
    J = torch.log(1+torch.exp(torch.neg(rnn_output)))
    
    J.backward()
    optimizer.step()
    
    return J.data.cpu().numpy()

###############################################################################################
strnn_model = STRNNModule().cuda()
optimizer = torch.optim.SGD(parameters(), lr=learning_rate, momentum=momentum)

for i in xrange(num_epochs):
    # Training
    total_loss = 0.
    train_batches = list(zip(train_time, train_lati, train_longi, train_loc))
    for j, train_batch in enumerate(train_batches):
        #inner_batches = data_loader.inner_iter(train_batch, batch_size)
        #for k, inner_batch in inner_batches:
        batch_time, batch_lati, batch_longi, batch_loc = train_batch#inner_batch)
        total_loss += run(train_user[j], batch_time, batch_lati, batch_longi, batch_loc, step=1)
        print j
        if (j+1) % 200 == 0:
            print("batch #{:d}: ".format(j+1)), "batch_loss :", total_loss/j, datetime.datetime.now()
    # Evaluation
    if (i+1) % evaluate_every == 0:
        print("==================================================================================")
        print("Evaluation at epoch #{:d}: ".format(i+1)), total_loss/j, datetime.datetime.now()
        valid_batches = list(zip(valid_time, valid_lati, valid_longi, valid_loc))
        print_score(valid_batches, step=2)

# Testing
print("Training End..")
print("==================================================================================")
print("Test: ")
test_batches = list(zip(test_time, test_lati, test_longi, test_loc))
print_score(test_batches, step=3)
