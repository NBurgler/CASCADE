import os
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import pandas as pd
import pickle
import sys
import gnn_layers

sys.path.append('code/predicting_model')

path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/Shift/DFTNN/"

def node_sets_fn(node_set, *, node_set_name):
    features = node_set.get_features_dict()

    if node_set_name == "atom":
        #atom_embedding = tf.keras.layers.Embedding(256) #Embedding on entire table of elements or just the organic atoms?
        return features
    
    return features
    
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
    if node_set_name == "atom":
        return tf.keras.layers.Embedding(35, 256)(node_set["atom_num"])
    return

def set_initial_edge_state(edge_set, *, edge_set_name):
    features = edge_set.get_features_dict()

    '''if edge_set_name == "bond":
        distances = features.pop('distance')
        features['rbf_distance'] = rbf_expansion(distances)'''
    return tf.keras.layers.Embedding(3, 256)(edge_set["bond_type"])

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


def build_model():
    batch_size = 32
    dataset = tf.data.TFRecordDataset(filenames=["data/own_data/shift_graph.tfrecords"])
    dataset = dataset.batch(batch_size)
    #TODO: train-test split

    graph_schema = tfgnn.read_schema("code/predicting_model/GraphSchema.pbtxt")
    graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    dataset = dataset.map(tfgnn.keras.layers.ParseExample(graph_tensor_spec))
    preproc_input_spec = dataset.element_spec


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

if __name__ == "__main__":
    model = build_model()
    model.summary()