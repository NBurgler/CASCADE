import tensorflow as tf
import tensorflow_gnn as tfgnn
import pandas as pd
import numpy as np

def rbf_expansion(distances, mu=0, delta=0.1, kmax=256):
    k = np.arange(0, kmax)
    logits = -(tf.expand_dims(distances, 1) - (-mu + delta * k))**2 / delta #Not sure if this is correct, but haven't found a way to check
    return tf.math.exp(logits)

def rbf_expansion_2(distances, mu=0, delta=0.1, kmax=256):
    k = np.arange(0, kmax)
    logits = -(np.atleast_2d(distances).T - (-mu + delta * k))**2 / delta
    return np.exp(logits)

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
    data = tf.data.TFRecordDataset(["data/own_data/easy_train.tfrecords"])
    graph_schema = tfgnn.read_schema("code/predicting_model/GraphSchema.pbtxt")
    graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    data = data.map(lambda serialized: tfgnn.parse_single_example(graph_tensor_spec, serialized))

    graph = data.take(1).get_single_element()

    bond_df = pd.read_csv("code/predicting_model/Shift/DFTNN/own_data_bond.csv.gz", index_col=0)

    graph = tfgnn.keras.layers.MapFeatures(edge_sets_fn=edge_sets_fn)(graph)
    print(graph.edge_sets['bond'].__getitem__('rbf_distance'))
    print(rbf_expansion_2(bond_df))

    #print(rbf_expansion_2(bond_df.loc[bond_df["mol_id"] == 21578]["distance"]))