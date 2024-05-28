import tensorflow as tf
import tensorflow_gnn as tfgnn
import pandas as pd
import numpy as np

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

def set_initial_edge_state(edge_set, *, edge_set_name):
    features = edge_set.get_features_dict()

    if edge_set_name == "bond":
        distances = features.pop('distance')
        features['rbf_distance'] = rbf_expansion(distances)
    
    return tf.keras.layers.Dense(256)(tf.expand_dims(tf.keras.layers.Concatenate()([edge_set["bond_type"], edge_set["rbf_distance"]]), axis=1))

if __name__ == "__main__":
    data = tf.data.TFRecordDataset(["data/own_data/shift_train.tfrecords"])
    graph_schema = tfgnn.read_schema("code/predicting_model/GraphSchema.pbtxt")
    graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    
    data = data.map(lambda serialized: tfgnn.parse_single_example(graph_tensor_spec, serialized))

    graph = data.take(1).get_single_element()
    print(graph)
    print(graph.edge_sets['bond'])
    print(graph.edge_sets['bond'].__getitem__('distance'))
    print(graph.edge_sets['bond'].__getitem__('bond_type'))
    print(graph.edge_sets['bond'].adjacency.source)
    print(graph.node_sets['atom'].__getitem__('atom_num'))

    bond_df = pd.read_csv("code/predicting_model/Shift/DFTNN/own_data_bond.csv.gz", index_col=0)