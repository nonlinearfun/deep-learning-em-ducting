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

from time import process_time
import time
import json
import pandas
import dataset
from dataset import *
import NeuralNet
from NeuralNet import *

import tensorflow as tf
from tensorflow import set_random_seed
set_random_seed(0)
from keras import models

def load_models(path, *modelpath):
    """ Return list of models from given filepaths """
    models = []
    for model in modelpath:
        filepath = path + str(model)
        models.append(NeuralNet(filepath=filepath))
    return models

def get_class_pred(class_models, X):
    """ Return class predictions for individual models & ensemble """
    probs, preds, evaltime = [], [], []
    for cNN in class_models:
        model = cNN.model
        prob = model.predict(X)
        pred = prob.argmax(axis=-1)
        probs.append(prob)
        preds.append(pred)
    ensemble_prob = (np.round(np.mean(probs, axis=0))).astype(int)
    ensemble_pred = ensemble_prob.argmax(axis=-1)
    preds.append(ensemble_pred)
    return preds

def calculate_indices(pred, cy):
    """ Return indices of correct evap & surface-based duct predictions """
    labels = np.argmax(cy, axis=1)
    correct = np.where(pred == labels)
    evap_preds = np.where(pred == 0)
    surf_preds = np.where(pred == 1)
    correct_evap_ind = np.intersect1d(evap_preds, correct)
    correct_surf_ind = np.intersect1d(surf_preds, correct)
    return correct_evap_ind, correct_surf_ind

def seperate_data(cpred, cy, X, ry_evap, ry_surf):
    """ Seperate data for evaporation and surface-based ducts """
    correct_evap_ind, correct_surf_ind = calculate_indices(cpred[0], cy)
    eX = X[correct_evap_ind,:]
    ery = ry_evap[correct_evap_ind,:]
    sX = X[correct_surf_ind,:]
    sry = ry_surf[correct_surf_ind-50,:]
    return eX, ery, sX, sry

def get_reg_pred(reg_models, X):
    """ Return regression predictions for individual models & ensemble """
    rmse, preds, evaltime = [], [], []
    for rNN in reg_models:
        model = rNN.model
        pred = model.predict(X)
        preds.append(pred)
    preds.append(np.mean(preds, axis=0))
    return preds

def evaluate_class_pred(cpred, cy):
    """ Return metrics from classification model evaluation """
    evap_metrics, surf_metrics = [], []
    for pred in cpred:
        correct_evap_ind, correct_surf_ind = calculate_indices(pred, cy)
        evap_metrics.append([len(correct_evap_ind)/50, len(correct_evap_ind)])
        surf_metrics.append([len(correct_surf_ind)/5000, len(correct_surf_ind)])
    return evap_metrics, surf_metrics

def calculate_rmse(pred, label):
    """ Calculate root mean squared error """
    return np.sqrt(np.mean(np.square(pred-label), axis=0))

def evaluate_reg_pred(rpred, ry):
    """ Return metrics from regression model evaluation """
    metrics = []
    for pred in rpred:
        metrics.append(list(calculate_rmse(pred, ry)))
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', action='store_true', help='perform ensemble')
    args = parser.parse_args()

    if args.ensemble:
        # load top five models for ensembling
        class_models = load_models('../models/class/20200501115034/', 53, 67, 17, 56, 21)
        evap_models = load_models('../models/evap/20200510104056/', 21, 25, 56, 72, 69)
        surf_models = load_models('../models/surf/20200504125006/', 23, 42, 37, 24, 35)
        filename = '../models/results/ensemble_eval_results.txt'
    else:
        # load only the top model
        class_models = load_models('../models/class/20200501115034/', 53)
        evap_models = load_models('../models/evap/20200510104056/', 21)
        surf_models = load_models('../models/surf/20200504125006/', 23)
        filename = '../models/results/single_eval_results.txt'

    # collection times for classification model, regression models, and two-step model
    ctime, rtime, ttime = [], [], []
    file = open(filename, "a")

    for beta in ['none', 'beta0', 'beta1']:

        # load test data
        X, cy, ry_evap, ry_surf = load_testing_data(beta)

        # perform classification
        starttime = process_time()
        cpred = get_class_pred(class_models, X)
        ctime.append(process_time() - starttime)

        # sort predictions into appropriate regression networks
        eX, ery, sX, sry = seperate_data(cpred, cy, X, ry_evap, ry_surf)

        # perform regression
        regstart = process_time()
        epreds = get_reg_pred(evap_models, eX)
        spreds = get_reg_pred(surf_models, sX)
        rtime.append(process_time() - regstart)
        ttime.append(process_time() - starttime)

        # evaluate predictions
        evap_metrics, surf_metrics = evaluate_class_pred(cpred, cy)
        ermse = evaluate_reg_pred(epreds, ery)
        eimprov = (np.array(ermse[0]) - np.array(ermse[-1]))/np.array(ermse[0])
        srmse = evaluate_reg_pred(spreds, sry)
        simprov = (np.array(srmse[0]) - np.array(srmse[-1]))/np.array(srmse[0])

        # write results to txt
        file.write("\n\nResults for " + beta+"\n")
        file.write("Classification\nevaporation metrics: ")
        json.dump(evap_metrics, file)
        file.write("\nsurface-based metrics: ")
        json.dump(surf_metrics, file)
        file.write("\nRegression\nevaporation rmse: ")
        json.dump(str(ermse), file)
        file.write("\nimprovement: ")
        json.dump(str(eimprov), file)
        file.write("\nsurface-based rmse: ")
        json.dump(str(srmse), file)
        file.write("\nimprovement: ")
        json.dump(str(simprov), file)

    # average time across noise colors and number of samples
    totaltime = np.mean(ttime)/5050
    classtime = np.mean(ctime)/5050
    regtime = np.mean(rtime)/5050
    file.write("\n\nTotal time: "+str(totaltime))
    file.write("\nClassification time: "+str(classtime))
    file.write("\nRegression time: "+str(regtime))
