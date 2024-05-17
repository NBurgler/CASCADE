import os
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import pandas as pd
import pickle
import sys
import gnn_layers

import wandb
from wandb.keras import WandbCallback

wandb.init(
        project="nmr-prediction",
        dir="/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/H/DFTNN",
        config={
        "learning_rate": 5E-4,
        "dataset": "cascade",
        "epochs": 10,
        "setting": "tfgnn",
    }
)

sys.path.append('code/predicting_model')
    
def rbf_expansion(distances, mu=0, delta=0.1, kmax=256):
    k = np.arange(0, kmax)
    logits = -(tf.expand_dims(distances, 1) - (-mu + delta * k))**2 / delta #Not sure if this is correct, but haven't found a way to check
    return tf.math.exp(logits)
    
def edge_sets_fn(edge_set, *, edge_set_name):
    features = edge_set.get_features_dict()

    if edge_set_name == "bond":
        #replace interatomic distance by the rbf distance
        distances = features.pop('distance')
        features['rbf_distance'] = rbf_expansion(distances)
    
    return features

def set_initial_node_state(node_set, *, node_set_name):
    # Since we only have one node set, we can ignore node_set_name.
    features = node_set.get_features_dict()

    if node_set_name == "atom":
        atom_type = features.pop("atom_symbol")
        print(atom_type)
        if atom_type == "H":
            index = 0
        elif atom_type == "C":
            index = 1
        elif atom_type == "O":
            index = 2
        elif atom_type == "N":
            index = 3

        one_hot = tf.one_hot(index, 4)
        features["atom_num"] = one_hot
        return tf.keras.layers.Embedding(4, 256)(node_set["atom_num"])
    
    return

def set_initial_edge_state(edge_set, *, edge_set_name):
    features = edge_set.get_features_dict()

    if edge_set_name == "bond":
        distances = features.pop('distance')
        features['rbf_distance'] = rbf_expansion(distances)
        return tf.keras.layers.Embedding(3, 256)(edge_set["bond_type"])
    
    return
    

def edge_updating():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation="softplus"),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(256, activation="softplus"),
        tf.keras.layers.Dense(256, activation="softplus")])

def node_updating():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="softplus"),
        tf.keras.layers.Dense(256)])


def build_model(preproc_input_spec):
    #preprocessing layers
    preproc_input = tf.keras.layers.Input(type_spec=preproc_input_spec)

    graph = preproc_input.merge_batch_to_components() #might need this

    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(graph)
    #TODO: no rbf expansion currently

    # Message passing layers
    # First, the edges are updated according to the layers in edge_updating
    # Then, the updated edge states are pooled to the node states (hadamard product + element-wise sum)
    # Finally, the pooled edge states are updated according to the layers in node_updating

    for _ in range(3):
        graph = tfgnn.keras.layers.GraphUpdate(
            edge_sets={"bond": tfgnn.keras.layers.EdgeSetUpdate(
                    next_state=tfgnn.keras.layers.ResidualNextState(
                        residual_block=edge_updating()
                )
            )},
            node_sets={"atom": tfgnn.keras.layers.NodeSetUpdate(
                {"bond": tfgnn.keras.layers.Pool(tag=tfgnn.SOURCE, reduce_type="prod|sum")},
                next_state=tfgnn.keras.layers.ResidualNextState(
                    residual_block=node_updating()
                )
            )}
        )(graph)

    readout_features = tfgnn.keras.layers.StructuredReadout("shift")(graph)
    logits = tf.keras.layers.Dense(256, activation="softplus")(readout_features)
    logits = tf.keras.layers.Dense(256, activation="softplus")(logits)
    logits = tf.keras.layers.Dense(128, activation="softplus")(logits)
    output = tf.keras.layers.Dense(1, activation="softplus")(logits)

    return tf.keras.Model(inputs=[preproc_input], outputs=[output])

def decode_fn(record_bytes):
  graph = tfgnn.parse_example(
      graph_tensor_spec, record_bytes, validate=True)

  # extract label from node and remove from input graph
  node_features = graph.node_sets["_readout"].get_features_dict()
  label = node_features.pop('shift')

  return graph, label


if __name__ == "__main__":
    path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/Shift/DFTNN/"
    batch_size = 5
    lr = 5E-4
    epochs = 10

    dataset = tf.data.TFRecordDataset(filenames=["data/own_data/shift_graph.tfrecords"])
    
    dataset = dataset.batch(batch_size)

    graph_schema = tfgnn.read_schema("code/predicting_model/GraphSchema.pbtxt")
    graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    dataset = dataset.map(decode_fn)
    preproc_input_spec, label_spec = dataset.element_spec

    dataset = dataset.shuffle(buffer_size=10)
    train_size = 70
    valid_size = 15
    test_size = 15

    train_ds = dataset.take(train_size)
    test_ds = dataset.skip(train_size)
    valid_ds = test_ds.take(valid_size)
    test_ds = test_ds.skip(test_size)

    model = build_model(preproc_input_spec)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='mae')
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(path, save_best_only=True, period=1, verbose=1)
    csv_logger = tf.keras.callbacks.CSVLogger('own_log.csv')

    def decay_fn(epoch, learning_rate):
        """ Jorgensen decays to 0.96*lr every 100,000 batches, which is approx
        every 28 epochs """
        if (epoch % 70) == 0:
            return 0.96 * learning_rate
        else:
            return learning_rate
        
    lr_decay = tf.keras.callbacks.LearningRateScheduler(decay_fn)

    hist = model.fit(train_ds, validation_data=valid_ds,
                    epochs=epochs, verbose=1, 
                    callbacks=[checkpoint, csv_logger, lr_decay, WandbCallback()])