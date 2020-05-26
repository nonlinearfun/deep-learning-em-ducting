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
import re
import pandas

from keras.utils import to_categorical

class dataset:
    """ get or create dataset object associated with filepath """
    def __init__(self, filepath):
        self.inputs, self.labels = None, None
        self.filepath = filepath
        self.process_data()

    def process_data(self):
        """ load data from numpy file (loads faster for hypersearch)
            if not found, create the numpy file from csv first """
        try:
            dataset = np.load(self.filepath+'.npz')
        except:
            df = pandas.read_csv(self.filepath+'.csv')
            data = df.values    # [num examples, 250]
            m,n = data.shape
            inputs, labels = [], []
            header = list(df)
            for ind, val in enumerate(header):
                inp = np.float32(data[:,ind])
                label = np.array(re.findall("\d+\.\d+",val.replace('_','.')), dtype=np.float32)
                inputs.append(inp)
                labels.append(label)
            np.savez(self.filepath+'.npz', X=inputs, y=labels)
            dataset = np.load(self.filepath+'.npz')
        self.inputs = dataset['X']
        ind, _ = self.inputs.shape
        self.labels = np.reshape(dataset['y'], [ind, -1])

def shuffle(*items):
    """ returns arguments shuffled together """
    index_array = np.arange(len(items[0]))
    np.random.shuffle(index_array)
    s_list = []
    for element in items:
        s_elem = []
        for value in index_array:
            s_elem.append(element[value])
        s_list.append(np.array(s_elem))
    return tuple(s_list)

def get_Xy(filepath):
    """ returns shuffled X/y from given filepath """
    data = dataset(filepath)
    X = data.inputs
    y = data.labels
    return shuffle(X, y)

def make_class_labels(evap, surf, train):
    """ create classification dataset with class labels"""
    n_c1 = len(evap)
    n_c2 = len(surf)
    if train:
        # equate evap and surf by upsampling for train and val sets
        c1_ind = np.random.choice(a=n_c1, size=n_c2)
        evap = evap[c1_ind,:]
        n_c1 = n_c2
    X = np.concatenate((evap, surf), axis=0)
    y = np.concatenate(([0]*n_c1, [1]*n_c2), axis=0)
    y = to_categorical(y, 2)
    if train:
        return shuffle(X, y)            # shuffle the train and val sets
    return X, y                         # don't need to shuffle test set

def load_training_reg(task):
    """ returns train & val data for regression task """
    try:
        datavar = np.load('../data/train/'+task+'.npy')
    except:
        datavar = {}
        for _, type in enumerate(('train', 'val')):
            filepath = '../data/train/'+task+type
            datavar['X_'+type], datavar['y_'+type] = get_Xy(filepath)
        np.save('../data/train/'+task+'.npy', datavar)
    return datavar

def load_training_class():
    """ returns train & val data for classification task """
    try:
        datavar = np.load('../data/train/class.npy')
    except:
        evap = load_training_reg('evap')
        surf = load_training_reg('surf')

        datavar = {}
        for _, type in enumerate(('train', 'val')):
            datavar['X_'+type], datavar['y_'+type] = make_class_labels(evap['X_'+type], surf['X_'+type], True)
        np.save('../data/train/class.npy', datavar)
    return datavar

def load_testing_data(noise):
    """ returns test data for specified noise """
    evap_path = '../data/test/evaptest'+noise
    surf_path = '../data/test/surftest'+noise
    X_evap, ry_evap = get_Xy(evap_path)
    X_surf, ry_surf = get_Xy(surf_path)
    X, y = make_class_labels(X_evap, X_surf, False)
    return X, y, ry_evap, ry_surf
