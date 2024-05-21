import os
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
import pandas as pd
import pickle
import sys
import functools

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
    '''if node_set_name == "atom":
        return tf.keras.layers.Embedding(4, 256)(node_set["atom_num"])'''
    return tf.keras.layers.Dense(256)(node_set["atom_num"])

def set_initial_edge_state(edge_set, *, edge_set_name):
    features = edge_set.get_features_dict()

    if edge_set_name == "bond":
        distances = features.pop('distance')
        features['rbf_distance'] = rbf_expansion(distances)
    
    return tf.keras.layers.Dense(256)(tf.expand_dims(tf.keras.layers.Concatenate()([edge_set["bond_type"], edge_set["rbf_distance"]]), axis=1))
    

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

    graph = preproc_input.merge_batch_to_components()

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
    initial_learning_rate = 5E-4
    epochs = 10
    epoch_divisor = 1

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


    task = runner.NodeMeanAbsoluteError("shift")
    steps_per_epoch = train_size // batch_size // epoch_divisor
    validation_steps = valid_size // batch_size // epoch_divisor
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, 70, 0.96
    )
    optimizer_fn = functools.partial(tf.keras.optimizers.Adam, learning_rate=learning_rate)

    trainer = runner.KerasTrainer(
        strategy=strategy,
        model_dir="/tmp/gnn_model/"
        callbacks=WandbCallback(),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        restore_best_weights=False,
        checkpoint_every_n_steps="never",
        summarize_every_n_steps="never",
        backup_and_restore=False,
    )