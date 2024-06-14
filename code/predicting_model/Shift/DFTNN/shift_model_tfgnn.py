import os
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
import pandas as pd
import pickle
import sys
import functools
import matplotlib.pyplot as plt
import datetime

import wandb
from wandb.keras import WandbCallback

'''wandb.init(
        project="nmr-prediction",
        dir="/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/H/DFTNN",
        config={
        "learning_rate": 5E-4,
        "dataset": "cascade",
        "epochs": 10,
        "setting": "tfgnn",
    }
)'''

sys.path.append('code/predicting_model')
    
def rbf_expansion(distances, mu=0, delta=0.1, kmax=256):
    k = np.arange(0, kmax)
    logits = -(tf.expand_dims(distances, 1) - (-mu + delta * k))**2 / delta
    return tf.math.exp(logits)
    
def edge_sets_fn(edge_set, *, edge_set_name):
    features = edge_set.get_features_dict()

    if edge_set_name == "bond":
        #replace interatomic distance by the rbf distance
        distances = features.pop('distance')
        features['rbf_distance'] = rbf_expansion(distances)
    
    return features

def set_initial_node_state(node_set, *, node_set_name):
    # one-hot encoded features are embedded immediately
    atom_num_embedding = tf.keras.layers.Dense(64)(node_set["atom_num"])
    chiral_tag_embedding = tf.keras.layers.Dense(64)(node_set["chiral_tag"])
    hybridization_embedding = tf.keras.layers.Dense(64)(node_set["hybridization"])

    # other numerical features are first reshaped...
    degree = tf.keras.layers.Reshape((-1, 1))(node_set["degree"])
    explicit_valence = tf.keras.layers.Reshape((-1, 1))(node_set["explicit_valence"])
    formal_charge = tf.keras.layers.Reshape((-1, 1))(node_set["formal_charge"])
    implicit_valence = tf.keras.layers.Reshape((-1, 1))(node_set["implicit_valence"])
    is_aromatic = tf.keras.layers.Reshape((-1, 1))(node_set["is_aromatic"])
    no_implicit = tf.keras.layers.Reshape((-1, 1))(node_set["no_implicit"])
    num_explicit_Hs = tf.keras.layers.Reshape((-1, 1))(node_set["num_explicit_Hs"])
    num_implicit_Hs = tf.keras.layers.Reshape((-1, 1))(node_set["num_implicit_Hs"])
    num_radical_electrons = tf.keras.layers.Reshape((-1, 1))(node_set["num_radical_electrons"])
    total_degree = tf.keras.layers.Reshape((-1, 1))(node_set["total_degree"])
    total_num_Hs = tf.keras.layers.Reshape((-1, 1))(node_set["total_num_Hs"])
    total_valence = tf.keras.layers.Reshape((-1, 1))(node_set["total_valence"])

    # ... and then concatenated so that they can be embedded as well
    numerical_features = tf.keras.layers.Concatenate(axis=2)([degree, explicit_valence, formal_charge, implicit_valence, is_aromatic, no_implicit,
                                                              num_explicit_Hs, num_implicit_Hs, num_radical_electrons, total_degree, total_num_Hs, 
                                                              total_valence])
    numerical_features = tf.squeeze(numerical_features, 1)
    numerical_embedding = tf.keras.layers.Dense(64)(numerical_features)
    
    # the one-hot and numerical embeddings are concatenated and fed to another dense layer
    concatenated_embedding = tf.keras.layers.Concatenate()([atom_num_embedding, chiral_tag_embedding,
                                                            hybridization_embedding, numerical_embedding])
    return tf.keras.layers.Dense(256)(concatenated_embedding)

def set_initial_edge_state(edge_set, *, edge_set_name):
    features = edge_set.get_features_dict()

    if edge_set_name == "bond":
        distances = features.pop('distance')
        features['rbf_distance'] = rbf_expansion(distances)
    # TODO: add other features
    return tf.keras.layers.Dense(256)(tf.expand_dims(tf.keras.layers.Concatenate()([edge_set["bond_type"], edge_set["rbf_distance"]]), axis=1))
    

def edge_updating():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu")])

def node_updating():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256)])

def readout_layers():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1)])


def model_fn(graph_tensor_spec: tfgnn.GraphTensorSpec):
    #preprocessing layers
    graph = inputs = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(graph)

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
                {"bond": tfgnn.keras.layers.Pool(tag=tfgnn.TARGET, reduce_type="prod|sum")},
                next_state=tfgnn.keras.layers.ResidualNextState(
                    residual_block=node_updating()
                )
            )}
        )(graph)

    #readout layers
    '''output = tfgnn.keras.layers.GraphUpdate(
        node_sets={"atom": tfgnn.keras.layers.NodeSetUpdate(
            {"bond": tfgnn.keras.layers.Pool(tag=tfgnn.SOURCE, reduce_type="sum")},
            next_state=tfgnn.keras.layers.NextStateFromConcat(
                transformation=readout_layers()
            )
        )}
    )(graph)'''
    output = graph

    '''readout_features = tfgnn.keras.layers.StructuredReadout("shift")(graph)
    logits = tf.keras.layers.Dense(256, activation="relu")(readout_features)
    logits = tf.keras.layers.Dense(256, activation="relu")(logits)
    logits = tf.keras.layers.Dense(128, activation="relu")(logits)
    output = tf.keras.layers.Dense(1)(logits)'''
    
    return tf.keras.Model(inputs=[inputs], outputs=[output])

def decode_fn(record_bytes):
    graph = tfgnn.parse_example(
        graph_tensor_spec, record_bytes, validate=True)

    # extract label from node and remove from input graph
    label = tfgnn.keras.layers.StructuredReadout("shift")(graph)
    graph = graph.remove_features(node_sets={"_readout": ["shift"]})

    return graph, label


if __name__ == "__main__":
    path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/Shift/DFTNN/"
    batch_size = 128
    initial_learning_rate = 5E-4
    epochs = 5
    epoch_divisor = 1

    train_ds_provider = runner.TFRecordDatasetProvider(filenames=["data/own_data/shift_train.tfrecords"])
    test_ds_provider = runner.TFRecordDatasetProvider(filenames=["data/own_data/shift_test.tfrecords"])
    valid_ds_provider = runner.TFRecordDatasetProvider(filenames=["data/own_data/shift_valid.tfrecords"])

    graph_schema = tfgnn.read_schema("code/predicting_model/GraphSchema.pbtxt")
    graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    train_size = 63565
    valid_size = 13622
    test_size = 13622

    task = runner.NodeMeanAbsoluteError("shift", label_feature_name="shift")
    steps_per_epoch = train_size // batch_size // epoch_divisor
    validation_steps = valid_size // batch_size // epoch_divisor
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    optimizer_fn = functools.partial(tf.keras.optimizers.Adam, learning_rate=learning_rate)

    filepath = path + "gnn/models/"
    log_dir = path + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, period=1, verbose=1)

    trainer = runner.KerasTrainer(
        strategy=tf.distribute.MirroredStrategy(),
        model_dir=path + "tmp/gnn_model/",
        callbacks=[tensorboard_callback, checkpoint],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        restore_best_weights=False,
        checkpoint_every_n_steps="never",
        summarize_every_n_steps="never",
        backup_and_restore=False,
    )

    model_exporter = runner.KerasModelExporter(output_names="shifts")

    runner.run(
        gtspec=graph_tensor_spec,
        train_ds_provider=train_ds_provider,
        valid_ds_provider=valid_ds_provider,
        model_fn=model_fn,
        optimizer_fn=optimizer_fn,
        trainer=trainer,
        task=task,
        global_batch_size=batch_size,
        epochs=epochs,
        model_exporters=[model_exporter]
    )