import sys
sys.path.append('code/predicting_model')
from collections import Counter
from sklearn.linear_model import LinearRegression

import os
import numpy as np
from numpy.random import seed
seed(1)
import array
import tensorflow
tensorflow.random.set_seed(2)

from nfp.preprocessing import MolPreprocessor, GraphSequence

import gzip
import pickle
import pandas as pd
import tensorflow as tf

# Define Keras model
from tensorflow import keras
import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from tensorflow.keras.layers import (Input, Embedding, Dense, BatchNormalization, Dropout,
                                 Concatenate, Multiply, Add, Reshape, Flatten)
from tensorflow.keras import models

from nfp.layers import (MessageLayer, GRUStep, Squeeze, EdgeNetwork,
                               ReduceAtomToMol, ReduceBondToAtom,
                               GatherAtomToBond, ReduceAtomToPro)
from nfp.models import GraphModel
import argparse

import wandb
from wandb.keras import WandbCallback


#wandb.init(
#        project="nmr-prediction",
#        dir="/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/Multiplicity",
#        config={
#        "learning_rate": 5E-4,
#        "dataset": "cascade",
#        "epochs": 1200,
#        "setting": "default",
#        "predction": "multiplicity",
#    }
#)

path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/Multiplicity/"

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--restart', action='store_true')
args = parser.parse_args()

train = pd.read_pickle(path + 'mult_train.pkl.gz')
valid = pd.read_pickle(path + 'mult_valid.pkl.gz')

y_train = train.Shape.values
y_valid = valid.Shape.values

def rbf_expansion(distances, mu=0, delta=0.1, kmax=256):
    k = np.arange(0, kmax)
    logits = -(np.atleast_2d(distances).T - (-mu + delta * k))**2 / delta
    return np.exp(logits)

def atomic_number_tokenizer(atom):
    return atom.GetNumRadicalElectrons()

def _compute_stacked_offsets(sizes, repeats):
    return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)

class RBFSequence(GraphSequence):
    def process_data(self, batch_data):
        batch_data['distance_rbf'] = rbf_expansion(batch_data['distance'])
        
        offset = _compute_stacked_offsets(
            batch_data['n_pro'], batch_data['n_atom'])

        offset = np.where(batch_data['atom_index']>=0, offset, 0)
        batch_data['atom_index'] += offset

        del batch_data['n_atom']
        del batch_data['n_bond']
        del batch_data['distance']

        return batch_data

with open(path + 'mult_processed_inputs.p', 'rb') as f:
    input_data = pickle.load(f)
    
preprocessor = input_data['preprocessor']

# Train a quick group-contribution model to get initial values for enthalpies per atom

#X = []
#Y = []
#for row,y in zip(input_data['inputs_train'], y_train): 
#    X.extend(row['atom'][row['atom_index']>=0])
#    Y.extend([[0, 1, 0, 0, 0, 0, 0, 0, 0],
#            [0, 0, 0, 0, 0, 0, 0, 0, 1],
#            [0, 0, 0, 0, 0, 0, 0, 0, 1],
#            [0, 0, 0, 0, 0, 0, 0, 0, 1],
#            [0, 0, 0, 0, 0, 0, 0, 0, 1],
#            [0, 0, 0, 0, 0, 0, 0, 0, 1],
#            [0, 0, 0, 0, 0, 0, 0, 0, 1]])

#atom_means = pd.DataFrame({'atom':X, 'shape':Y}).dropna().groupby('atom')['shape']

#atom_means = atom_means.reindex(np.arange(preprocessor.atom_classes)).fillna(0)

# Construct input sequences
batch_size = 32

train_sequence = RBFSequence(input_data['inputs_train'], y_train, batch_size)
valid_sequence = RBFSequence(input_data['inputs_valid'], y_valid, batch_size)
#print(train_sequence._inputs[0])
#print(train_sequence._y[0])

#print(input_data['inputs_train'])

# Raw (integer) graph inputs
atom_index = Input(shape=(1,), name='atom_index', dtype='int32')
atom_types = Input(shape=(1,), name='atom', dtype='int32')
distance_rbf = Input(shape=(256,), name='distance_rbf', dtype='float32')
connectivity = Input(shape=(2,), name='connectivity', dtype='int32')
n_pro = Input(shape=(1,), name='n_pro', dtype='int32')

squeeze = Squeeze()

satom_index = squeeze(atom_index)
satom_types = squeeze(atom_types)
sn_pro = squeeze(n_pro)
# Initialize RNN and MessageLayer instances
atom_features = 256

# Initialize the atom states
atom_state = Embedding(
    preprocessor.atom_classes,
    atom_features, name='atom_embedding')(satom_types)

atomwise_shape = Embedding(
    preprocessor.atom_classes, 1, name='atomwise_shape',
    embeddings_initializer=keras.initializers.constant(1)   #TODO check initialization
)(satom_types)

bond_state = distance_rbf

def message_block(atom_state, bond_state, connectivity):

    atom_state = Dense(atom_features, use_bias=False)(atom_state)

    source_atom_gather = GatherAtomToBond(1)
    target_atom_gather = GatherAtomToBond(0)

    source_atom = source_atom_gather([atom_state, connectivity])
    target_atom = target_atom_gather([atom_state, connectivity])

    # Edge update network
    bond_state_message = Concatenate()([source_atom, target_atom, bond_state])
    bond_state_message = Dense(2*atom_features, activation='softplus')(bond_state_message)
    bond_state_message = Dense(atom_features)(bond_state_message)

    bond_state_message = Dense(atom_features, activation='softplus')(bond_state_message)
    bond_state_message = Dense(atom_features, activation='softplus')(bond_state_message)
    bond_state = Add()([bond_state_message, bond_state])

    # message function
    messages = Multiply()([source_atom, bond_state])
    messages = ReduceBondToAtom(reducer='sum')([messages, connectivity])
    
    # state transition function
    messages = Dense(atom_features, activation='softplus')(messages)
    messages = Dense(atom_features)(messages)
    atom_state = Add()([atom_state, messages])
    
    return atom_state, bond_state

for _ in range(3):
    atom_state, bond_state = message_block(atom_state, bond_state, connectivity)
    

atom_state = ReduceAtomToPro(reducer='unsorted_mean')([atom_state, satom_index, sn_pro])
atomwise_shape = ReduceAtomToPro(reducer='unsorted_mean')([atomwise_shape, satom_index, sn_pro])

atom_state = Dense(atom_features, activation='softplus')(atom_state)
atom_state = Dense(atom_features, activation='softplus')(atom_state)
atom_state = Dense(atom_features//2, activation='softplus')(atom_state)
atom_state = Dense(6 * 9, activation='softplus')(atom_state)
atom_state = Reshape((6, 9))(atom_state)
atom_state = Dense(9, activation='softmax')(atom_state)

output = Add()([atom_state, atomwise_shape])
output = Flatten()(output)

filepath = "multiplicity_model.hdf5"

lr = 5E-4
epochs = 1200

if args.restart:
    model = load_model(filepath, custom_objects={'GraphModel': GraphModel, 
                                                 'Squeeze': Squeeze,
                                                 'GatherAtomToBond': GatherAtomToBond,
                                                 'ReduceBondToAtom': ReduceBondToAtom,
                                                 'ReduceAtomToPro': ReduceAtomToPro})
else:
    model = GraphModel([
        atom_index, atom_types, distance_rbf, connectivity, n_pro], [output])

    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='cce')

model.summary()
    
checkpoint = ModelCheckpoint(filepath, save_best_only=True, period=1, verbose=1)
csv_logger = CSVLogger('own_log.csv')

def decay_fn(epoch, learning_rate):
    """ Jorgensen decays to 0.96*lr every 100,000 batches, which is approx
    every 28 epochs """
    if (epoch % 70) == 0:
        return 0.96 * learning_rate
    else:
        return learning_rate

#intermediate_layer_model = Model(inputs=model.input,
#                                 outputs=model.get_layer('reduce_atom_to_pro_2').output)
#data = train_sequence[0][0]
#intermediate_out = intermediate_layer_model.predict_on_batch(data)
#print(intermediate_out)
    
#print(train_sequence._y)

lr_decay = LearningRateScheduler(decay_fn)

hist = model.fit(train_sequence, validation_data=valid_sequence,
                 epochs=epochs, verbose=1, 
                 callbacks=[checkpoint, csv_logger, lr_decay])
