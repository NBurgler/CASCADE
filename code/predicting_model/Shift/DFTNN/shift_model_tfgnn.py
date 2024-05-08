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
    for _ in range(3):
        node_message = tfgnn.keras.layers.GraphUpdate(
            node_sets={"atom": tfgnn.keras.layers.NodeSetUpdate(
                {"bond": tfgnn.keras.layers.SimpleConv(tf.keras.layers.Dense(256), "mean")},
                next_state=tfgnn.keras.layers.NextStateFromConcat(tf.keras.layers.Dense(256, use_bias=False)))
            }
        )(graph)

        # Edge Updating
        # Turn into graph updates
        edge_message = tfgnn.keras.layers.GraphUpdate(
                edge_sets={"bond": tfgnn.keras.layers.EdgeSetUpdate(
                    next_state=tfgnn.keras.layers.NextStateFromConcat(
                        transformation=edge_updating()
                )
            )}
        )(node_message)
        
        #edge_message = tf.keras.layers.Add()([graph.edge_sets["bond"].__getitem__(tfgnn.HIDDEN_STATE), edge_message.edge_sets["bond"].__getitem__(tfgnn.HIDDEN_STATE)])
        edge_message = tfgnn.combine_values([graph.edge_sets["bond"].__getitem__(tfgnn.HIDDEN_STATE), 
                                            edge_message.edge_sets["bond"].__getitem__(tfgnn.HIDDEN_STATE)], combine_type="sum")

        graph_message = tf.keras.layers.Multiply()([node_message.node_sets["atom"].__getitem__(tfgnn.HIDDEN_STATE), edge_message])

        graph_message = tfgnn.pool_neighbors_to_node(graph_message, edge_set_name="bond", to_tag=tfgnn.SOURCE, reduce_type="sum", feature_name=tfgnn.HIDDEN_STATE)

        graph_message = tfgnn.keras.layers.GraphUpdate(
            node_sets={"atom": tfgnn.keras.layers.NodeSetUpdate(
                {"bond": tfgnn.keras.layers.SimpleConv(tf.keras.layers.Dense(256), "mean")},
                next_state=node_updating())
            }
        )(graph_message)

        graph = tf.keras.layers.Add()([graph, graph_message])

    readout_features = tfgnn.keras.layers.Pool(
        tfgnn.NODES, "mean", node_set_name="atom")(graph)
    readout_features = tfgnn.keras.layers.StructuredReadout(readout_features)
    logits = tf.keras.layers.Dense(256, activation="softplus")(readout_features)
    logits = tf.keras.layers.Dense(256, activation="softplus")(logits)
    logits = tf.keras.layers.Dense(128, activation="softplus")(logits)
    output = tf.keras.layers.Dense(1, activation="softplus")(logits)

    return tf.keras.Model(inputs=[preproc_input], outputs=[output])

if __name__ == "__main__":
    model = build_model()
    print(model)