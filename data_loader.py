import numpy as np
from datetime import datetime
import pandas as pd

def treat_prepro(train, step):
    train_f = open(train, 'r')
    # Need to change depending on threshold
    if step==1:
        lines = train_f.readlines()#[:86445] #659 #[:309931]
    elif step==2:
        lines = train_f.readlines()#[:13505]#[:309931]
    elif step==3:
        lines = train_f.readlines()#[:30622]#[:309931]

    train_user = []
    train_td = []
    train_ld = []
    train_loc = []
    train_dst = []

    user = 1
    user_td = []
    user_ld = []
    user_loc = []
    user_dst = []

    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        if len(tokens) < 3:
            if user_td: 
                train_user.append(user)
                train_td.append(user_td)
                train_ld.append(user_ld)
                train_loc.append(user_loc)
                train_dst.append(user_dst)
            user = int(tokens[0])
            user_td = []
            user_ld = []
            user_loc = []
            user_dst = []
            continue
        td = np.array([float(t) for t in tokens[0].split(',')])
        ld = np.array([float(t) for t in tokens[1].split(',')])
        loc = np.array([int(t) for t in tokens[2].split(',')])
        dst = int(tokens[3])
        user_td.append(td)
        user_ld.append(ld)
        user_loc.append(loc)
        user_dst.append(dst)

    if user_td: 
        train_user.append(user)
        train_td.append(user_td)
        train_ld.append(user_ld)
        train_loc.append(user_loc)
        train_dst.append(user_dst)

    return train_user, train_td, train_ld, train_loc, train_dst

def load_data(train):
    user2id = {}
    poi2id = {}

    train_user = []
    train_time = []
    train_lati = []
    train_longi = []
    train_loc = []
    valid_user = []
    valid_time = []
    valid_lati = []
    valid_longi = []
    valid_loc = []
    test_user = []
    test_time = []
    test_lati = []
    test_longi = []
    test_loc = []

    train_f = open(train, 'r')
    lines = train_f.readlines()

    user_time = []
    user_lati = []
    user_longi = []
    user_loc = []
    visit_thr = 30

    prev_user = int(lines[0].split('\t')[0])
    visit_cnt = 0
    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        user = int(tokens[0])
        if user==prev_user:
            visit_cnt += 1
        else:
            if visit_cnt >= visit_thr:
                user2id[prev_user] = len(user2id)
            prev_user = user
            visit_cnt = 1

    train_f = open(train, 'r')
    lines = train_f.readlines()

    prev_user = int(lines[0].split('\t')[0])
    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        user = int(tokens[0])
        if user2id.get(user) is None:
            continue
        user = user2id.get(user)

        time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")\
                -datetime(2009,1,1)).total_seconds()/60  # minutes
        lati = float(tokens[2])
        longi = float(tokens[3])
        location = int(tokens[4])
        if poi2id.get(location) is None:
            poi2id[location] = len(poi2id)
        location = poi2id.get(location)

        if user == prev_user:
            user_time.insert(0, time)
            user_lati.insert(0, lati)
            user_longi.insert(0, longi)
            user_loc.insert(0, location)
        else:
            train_thr = int(len(user_time) * 0.7)
            valid_thr = int(len(user_time) * 0.8)
            train_user.append(user)
            train_time.append(user_time[:train_thr])
            train_lati.append(user_lati[:train_thr])
            train_longi.append(user_longi[:train_thr])
            train_loc.append(user_loc[:train_thr])
            valid_user.append(user)
            valid_time.append(user_time[train_thr:valid_thr])
            valid_lati.append(user_lati[train_thr:valid_thr])
            valid_longi.append(user_longi[train_thr:valid_thr])
            valid_loc.append(user_loc[train_thr:valid_thr])
            test_user.append(user)
            test_time.append(user_time[valid_thr:])
            test_lati.append(user_lati[valid_thr:])
            test_longi.append(user_longi[valid_thr:])
            test_loc.append(user_loc[valid_thr:])

            prev_user = user
            user_time = [time]
            user_lati = [lati]
            user_longi = [longi]
            user_loc = [location]

    if user2id.get(user) is not None:
        train_thr = int(len(user_time) * 0.7)
        valid_thr = int(len(user_time) * 0.8)
        train_user.append(user)
        train_time.append(user_time[:train_thr])
        train_lati.append(user_lati[:train_thr])
        train_longi.append(user_longi[:train_thr])
        train_loc.append(user_loc[:train_thr])
        valid_user.append(user)
        valid_time.append(user_time[train_thr:valid_thr])
        valid_lati.append(user_lati[train_thr:valid_thr])
        valid_longi.append(user_longi[train_thr:valid_thr])
        valid_loc.append(user_loc[train_thr:valid_thr])
        test_user.append(user)
        test_time.append(user_time[valid_thr:])
        test_lati.append(user_lati[valid_thr:])
        test_loc.append(user_loc[valid_thr:])

    return len(user2id), poi2id, train_user, train_time, train_lati, train_longi, train_loc, valid_user, valid_time, valid_lati, valid_longi, valid_loc, test_user, test_time, test_lati, test_longi, test_loc

def inner_iter(data, batch_size):
    data_size = len(data)
    num_batches = int(len(data)/batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]
