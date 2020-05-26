# Copyright (C) 2020 Hilarie Sit
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
np.random.seed(0)
import os
os.environ['PYTHONHASHSEED']=str(0)
import random
random.seed(0)
import tensorflow as tf
from tensorflow import set_random_seed
set_random_seed(0)

import time
import pandas
import dataset
import multiprocessing
import threading

from dataset import *
import NeuralNet
from NeuralNet import *

def worker(task, datavar, grid, cparams, model_id, q_out):
    """ Have worker perform select number of search instances """
    newgrid = []
    # iterate through grid (list of vparams)
    for vparams in grid:
        filename = FILENAME+'/'+str(model_id)
        os.makedirs(filename)
        vparams['filepath'] = str(model_id)

        # save param info as txt file
        info_file = open(filename+"/parameters.txt","w+")
        info_file.write(str(vparams))
        info_file.write(str(cparams))
        info_file.close()

        # train
        start_time = time.time()
        NNmodel = NeuralNet(filename, datavar=datavar, vparams=vparams, cparams=cparams)
        end_time = time.time() - start_time

        # evaluate on training and validation sets
        model = NNmodel.model
        training_metric = model.evaluate(datavar['X_train'], datavar['y_train'])
        val_metric = model.evaluate(datavar['X_val'], datavar['y_val'])
        if task != 'class':
            training_metric[1] = np.sqrt(training_metric[1])
            val_metric[1] = np.sqrt(val_metric[1])
        vparams['training metric'] = training_metric
        vparams['val metric'] = val_metric
        vparams['training time'] = end_time
        newgrid.append(vparams)
        model_id+=1

    # return vparams with all info about model in newgrid
    q_out.put(newgrid)

def model_selection(task, cparams, vparams, datavar):
    """ Split up hyperparameter search among CPU threads """
    n_model = len(list(grid))                   # total number of models
    n_threads = multiprocessing.cpu_count()     # max number of threads
    mpt = int(np.ceil(n_model/n_threads))       # models per thread

    # split models among workers to parallelize hyperparameter search
    out_q = multiprocessing.Queue()
    for i in range(n_threads):
        if i == n_threads - 1:                  # this is the last thread!
            p = multiprocessing.Process(target=worker, args=(task, datavar, list(grid)[i*mpt:], cparams, i*mpt+1, out_q))
        else:
            p = multiprocessing.Process(target=worker, args=(task, datavar, list(grid)[i*mpt:i*mpt+mpt], cparams, i*mpt+1, out_q))
        p.start()
    p.join()

    # return vparams from all workers
    newgrid = []
    for i in range(n_threads):
        newgrid = newgrid + out_q.get()
    return newgrid

def get_vparams(bs_low, bs_high, drop_high, out_neuron):
    h_layers = np.random.randint(4,7)
    if h_layers == 4:
        neurons = [np.random.randint(800, 1001), np.random.randint(600, 801), np.random.randint(400, 601), np.random.randint(200, 401), out_neuron]
    elif h_layers == 5:
        neurons = [np.random.randint(800, 1001), np.random.randint(600, 801), np.random.randint(400, 601), np.random.randint(200, 401), np.random.randint(200, 401), out_neuron]
    else:
        neurons = [np.random.randint(800, 1001), np.random.randint(600, 801), np.random.randint(400, 601), np.random.randint(200, 401), np.random.randint(200, 401), np.random.randint(200, 401), out_neuron]
    dropout = random.uniform(0.2, drop_high)
    bs = np.random.randint(bs_low, bs_high)
    lr = random.uniform(1e-4, 1e-3)
    return neurons, dropout, lr, bs


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    # define parser arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='surf', help='which task')
    args = parser.parse_args()

    # Set basepath for all files created in this hyperparam search
    timestamp = time.strftime("%Y%m%d%H%M%S")
    FILENAME = '../models/'+args.task+'/'+timestamp
    os.makedirs(FILENAME)

    n_threads = multiprocessing.cpu_count()     # max number of threads
    n_models = n_threads*3                      # total number of models
    grid = []                                   # initialize grid for vparams

    if args.task == 'class':
        datavar = load_training_class()
        cparams = {
            'epochs': 40,
            'loss': 'binary_crossentropy',
            'activations': 'relu',
            'last_act': 'softmax',
            'metrics': ['accuracy', 'val_accuracy', 'max']}
        bs_low = 50
        bs_high = 201
        drop_high = 0.8
        out_neuron = 2

    if args.task == 'evap':
        datavar = load_training_reg(args.task)
        cparams = {
            'epochs': 500,
            'loss': 'mean_squared_error',
            'activations': 'relu',
            'last_act': 'linear',
            'metrics': ['mse', 'val_loss', 'min']}
        bs_low = 30
        bs_high = 101
        drop_high = 0.8
        out_neuron = 1

    if args.task == 'surf':
        n_models = n_threads*2
        datavar = load_training_reg(args.task)
        # constant variables
        cparams = {
            'epochs': 3000,
            'activations': 'relu',
            'last_act': 'linear',
            'loss': 'mean_squared_error',
            'metrics': ['mse', 'val_loss', 'min']}
        bs_low = 50
        bs_high = 201
        drop_high = 0.4
        out_neuron = 3

    for i in range(n_models):
        neurons, dropout, lr, bs = get_vparams(bs_low, bs_high, drop_high, out_neuron)
        vparams = {'neurons': neurons, 'lr': lr, 'bs': bs, 'dropout': dropout}
        grid.append(vparams)

    def save_table(*dicts, header, title):
        global MESSAGE
        dicts = np.array(dicts)
        comb_dict = {k: list(map(lambda d: d.get(k), dicts)) for k in set().union(*dicts)}
        df = pd.DataFrame(data=comb_dict, index=header).transpose()
        df.to_csv('../models/'+args.task+'/'+args.task+title+'.csv')

    save_table(cparams, header=['Constant parameters'], title='Constant')
    newgrid = model_selection(args.task, cparams, grid, datavar)
    save_table(*newgrid, header=pd.RangeIndex(1, n_models+1), title='Variable')
