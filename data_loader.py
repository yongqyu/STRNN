import numpy as np
from datetime import datetime
import random
import pandas as pd

def load_data(train):
    user2id = {}
    poi2id = {}
    poi2pos = {}

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
    prev_user = int(lines[0].split('\t')[0])
    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        user = int(tokens[0])
        time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")-datetime(2009,1,1)).total_seconds()/60  # minutes
        lati = float(tokens[2])
        longi = float(tokens[3])
        location = int(tokens[4])
        if user2id.get(user) is None:
            user2id[user] = len(user2id)
        user = user2id.get(user)
        if poi2id.get(location) is None:
            poi2id[location] = len(poi2id)
            poi2pos[poi2id[location]] = (lati, longi)
        location = poi2id.get(location)

        if user is prev_user:
            user_time.insert(0, time)
            user_lati.insert(0, lati)
            user_longi.insert(0, longi)
            user_loc.insert(0, location)
        else:
            train_thr = int(len(user_time) * 0.7)
            valid_thr = int(len(user_time) * 0.8)
            if train_thr > 35: 
                train_user.append(user)
                train_time.append(user_time[:train_thr])
                train_lati.append(user_lati[:train_thr])
                train_longi.append(user_longi[:train_thr])
                train_loc.append(user_loc[:train_thr])
                valid_user.append(user)
                valid_time.append(user_time[:valid_thr])
                valid_lati.append(user_lati[:valid_thr])
                valid_longi.append(user_longi[:valid_thr])
                valid_loc.append(user_loc[:valid_thr])
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

    train_thr = int(len(user_time) * 0.7)
    valid_thr = int(len(user_time) * 0.8)
    if train_thr > 35:
        trian_user.append(user)
        train_time.append(user_time[:train_thr])
        train_lati.append(user_lati[:train_thr])
        train_longi.append(user_longi[:train_thr])
        train_loc.append(user_loc[:train_thr])
        valid_user.append(user)
        valid_time.append(user_time[:valid_thr])
        valid_lati.append(user_lati[:valid_thr])
        valid_longi.append(user_longi[:valid_thr])
        valid_loc.append(user_loc[:valid_thr])
        test_user.append(user)
        test_time.append(user_time[valid_thr:])
        test_lati.append(user_lati[valid_thr:])
        test_loc.append(user_loc[valid_thr:])

    return len(user2id), poi2pos, train_user, train_time, train_lati, train_longi, train_loc, valid_user, valid_time, valid_lati, valid_longi, valid_loc, test_user, test_time, test_lati, test_longi, test_loc

def inner_iter(data, batch_size):
    data_size = len(data)
    num_batches = int(len(data)/batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]
