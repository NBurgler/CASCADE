import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import itertools
import pandas as pd
from tensorflow_gnn import runner

def check_zero(input_graph):
    i = 0
    for item in input_graph.node_sets["_readout"].__getitem__("shift"):
        if(0 in item):
            print(i)
            print(item)
        i += 1

    #print(input_graph.context.__getitem__("smiles")[28872])

def rbf_expansion(distances, mu=0, delta=0.1, kmax=256):
    k = np.arange(0, kmax)
    logits = -(tf.expand_dims(distances, 1) - (-mu + delta * k))**2 / delta
    return tf.math.exp(logits)

def set_initial_node_state(node_set, *, node_set_name):
    # one-hot encoded features are embedded immediately
    atom_num_embedding = tf.keras.layers.Dense(64)(node_set["atom_num"])
    chiral_tag_embedding = tf.keras.layers.Dense(64)(node_set["chiral_tag"])
    hybridization_embedding = tf.keras.layers.Dense(64)(node_set["hybridization"])

    # other numerical features are first reshaped...
    degree = tf.keras.layers.Reshape((-1,))(node_set["degree"])
    explicit_valence = tf.keras.layers.Reshape((-1,))(node_set["explicit_valence"])
    formal_charge = tf.keras.layers.Reshape((-1,))(node_set["formal_charge"])
    implicit_valence = tf.keras.layers.Reshape((-1,))(node_set["implicit_valence"])
    is_aromatic = tf.keras.layers.Reshape((-1,))(node_set["is_aromatic"])
    no_implicit = tf.keras.layers.Reshape((-1,))(node_set["no_implicit"])
    num_explicit_Hs = tf.keras.layers.Reshape((-1,))(node_set["num_explicit_Hs"])
    num_implicit_Hs = tf.keras.layers.Reshape((-1,))(node_set["num_implicit_Hs"])
    num_radical_electrons = tf.keras.layers.Reshape((-1,))(node_set["num_radical_electrons"])
    total_degree = tf.keras.layers.Reshape((-1,))(node_set["total_degree"])
    total_num_Hs = tf.keras.layers.Reshape((-1,))(node_set["total_num_Hs"])
    total_valence = tf.keras.layers.Reshape((-1,))(node_set["total_valence"])

    print(node_set["degree"])
    print(degree)

    # ... and then concatenated so that they can be embedded as well
    numerical_features = tf.keras.layers.Concatenate(axis=1)([degree, explicit_valence, formal_charge, implicit_valence, is_aromatic, no_implicit,
                                                              num_explicit_Hs, num_implicit_Hs, num_radical_electrons, total_degree, total_num_Hs, 
                                                              total_valence])
    print(numerical_features)
    #numerical_features = tf.squeeze(numerical_features, 1)
    #print(numerical_features)
    numerical_embedding = tf.keras.layers.Dense(64)(numerical_features)
    
    # the one-hot and numerical embeddings are concatenated and fed to another dense layer
    concatenated_embedding = tf.keras.layers.Concatenate()([atom_num_embedding, chiral_tag_embedding,
                                                            hybridization_embedding, numerical_embedding])
    return tf.keras.layers.Dense(256, name="node_init")(concatenated_embedding)

def set_initial_edge_state(edge_set, *, edge_set_name):
    features = edge_set.get_features_dict()

    if edge_set_name == "bond":
        distances = features.pop('distance')
        features['rbf_distance'] = rbf_expansion(distances)
    # TODO: add other features
    return tf.keras.layers.Dense(256, name="edge_init")(features['rbf_distance'])


if __name__ == "__main__":
    path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/Shift/DFTNN/"

    graph_schema = tfgnn.read_schema("code/predicting_model/GraphSchema.pbtxt")
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    model = tf.saved_model.load(path + "tmp/gnn_model/export")
    signature_fn = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    num_examples = 2
    dataset_provider = runner.TFRecordDatasetProvider(filenames=["data/own_data/shift_train.tfrecords"])
    dataset = dataset_provider.get_dataset(tf.distribute.InputContext())

    example = next(iter(dataset.batch(1)))
    input_graph = tfgnn.parse_example(graph_spec, example)
    graph = input_graph
    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(graph)
    print(graph)
    
    check_zero(input_graph)

    train_df = pd.read_pickle('code/predicting_model/Shift/DFTNN/cascade_train.pkl.gz', compression="gzip")
    print(train_df)

    input_dict = {"examples": example}
    output_dict = signature_fn(**input_dict)
    logits = output_dict["shifts"]

    #print(logits)