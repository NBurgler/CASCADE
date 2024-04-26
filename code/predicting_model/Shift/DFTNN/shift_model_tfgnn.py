import os
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import pandas as pd
import pickle
import sys

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


if __name__ == "__main__":
    batch_size = 32
    dataset = tf.data.TFRecordDataset(filenames=["data/own_data/shift_graph.tfrecords"])
    dataset = dataset.batch(batch_size)
    #TODO: train-test split

    graph_schema = tfgnn.read_schema("code/predicting_model/GraphSchema.pbtxt")
    graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    dataset = dataset.map(tfgnn.keras.layers.ParseExample(graph_tensor_spec))
    preproc_input_spec = dataset.element_spec

    preproc_input = tf.keras.layers.Input(type_spec=preproc_input_spec)
    tf.keras.backend.print_tensor(preproc_input.edge_sets['bond'].__getitem__('distance'))
    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn, edge_sets_fn=edge_sets_fn)(preproc_input)
    tf.keras.backend.print_tensor(graph.edge_sets['bond'].__getitem__('rbf_distance'))
