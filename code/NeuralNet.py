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

import pandas as pd

class NeuralNet:
    """ Returns one instance of trained neural network on provided data;
        if model exists on filepath, load model and history instead
    """
    def __init__(self, filepath, datavar=None, vparams=None, cparams=None):
        try:
            from keras import models
            self.model = models.load_model(filepath+'/model.h5')
            self.history = pd.read_csv(filepath+'/training.log')
        except:
            # initialize network parameters
            self.datavar = datavar
            self.loss, self.metrics, self.epochs = cparams['loss'], cparams['metrics'], cparams['epochs']
            self.neurons, self.dropout = vparams['neurons'], vparams['dropout']
            self.lr, self.bs = vparams['lr'], vparams['bs']
            self.act = [cparams['activations']]*(len(self.neurons)-1)+[cparams['last_act']]
            self.filepath = filepath
            # train network
            self.model, self.history = self.train()

    def train(self):
        """ Train neural network on one thread and save history and best weights """
        from keras import layers
        from keras import models
        from keras import optimizers
        from keras import callbacks
        import keras.backend as K
        import tensorflow as tf

        # force tensorflow to use a single thread
        session_conf = tf.ConfigProto(
              intra_op_parallelism_threads=1,
              inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

        # neural network architecture
        [_, dim] = self.datavar['X_train'].shape
        h_layers = len(self.neurons)-1
        model = models.Sequential()
        model.add(layers.Dense(self.neurons[0], activation=self.act[0], input_dim=dim))
        for h in range(h_layers):
            if self.dropout != None:
                model.add(layers.normalization.BatchNormalization())
                model.add(layers.Dropout(rate=self.dropout))
            model.add(layers.Dense(self.neurons[h+1], activation=self.act[h+1]))
        opt = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss=self.loss, optimizer=opt, metrics=[self.metrics[0]])

        # save training history
        csv_logger = callbacks.CSVLogger(self.filepath+'/training.log')
        # save model checkpoints
        checkpoint = callbacks.ModelCheckpoint(self.filepath+'/model.h5',
            monitor=self.metrics[1], mode=self.metrics[2], verbose=2, save_best_only=True)

        # TRAIN
        model.fit(self.datavar['X_train'], self.datavar['y_train'], validation_data=[self.datavar['X_val'],
            self.datavar['y_val']], epochs=self.epochs, batch_size=self.bs, verbose=2, callbacks=[checkpoint, csv_logger])
        model = models.load_model(self.filepath+'/model.h5')
        history = pd.read_csv(self.filepath+'/training.log')
        return model, history
