import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np
import sys
import os
sys.path.append('code')
from predicting_model.create_graph_tensor import create_dictionary, create_single_tensor

def convert_shape_one_hot(model_outputs):
    shapes = []
    for one_hot_matrix in model_outputs:
        shape = ''
        for one_hot in one_hot_matrix:
            if np.argmax(one_hot) == 0: shape += "m"
            elif np.argmax(one_hot) == 1: shape += "s"
            elif np.argmax(one_hot) == 2: shape += "d"
            elif np.argmax(one_hot) == 3: shape += "t"
            elif np.argmax(one_hot) == 4: shape += "q"
            elif np.argmax(one_hot) == 5: shape += "p"
            elif np.argmax(one_hot) == 6: shape += "h"
            elif np.argmax(one_hot) == 7: shape += "v"
        shapes.append(shape)

    return shapes

def predict_shift(path, data):
    graph_schema = tfgnn.read_schema(path + "code/predicting_model/GraphSchema.pbtxt")
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    shift_model = tf.saved_model.load(path + "code/predicting_model/Shift/gnn/models/test_model")
    signature_fn = shift_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    example = tf.reshape(data, (1,))
    input_graph = tfgnn.parse_example(graph_spec, example)
    input_dict = {"examples": example}
    output_dict = signature_fn(**input_dict)
    logits = output_dict["shifts"]
    return logits.numpy()

def predict_shape(path, data):
    graph_schema = tfgnn.read_schema(path + "code/predicting_model/GraphSchemaMult.pbtxt")
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    shape_model = tf.saved_model.load(path + "code/predicting_model/Shape/gnn/models/shape_model_best")
    signature_fn = shape_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    example = tf.reshape(data, (1,))
    input_graph = tfgnn.parse_example(graph_spec, example)
    input_dict = {"examples": example}
    output_dict = signature_fn(**input_dict)
    logits = output_dict["shape"]
    shapes = convert_shape_one_hot(logits)
    shapes = [shape.rstrip("s") for shape in shapes]
    shapes = ["s" if shape == '' else shape for shape in shapes]

    return shapes

def predict_coupling(path, data):
    graph_schema = tfgnn.read_schema(path + "code/predicting_model/GraphSchemaCoupling.pbtxt")
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    coupling_model = tf.saved_model.load(path + "code/predicting_model/Coupling/gnn/models/coupling_model_best")
    signature_fn = coupling_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    example = tf.reshape(data, (1,))
    input_graph = tfgnn.parse_example(graph_spec, example)
    input_dict = {"examples": example}
    output_dict = signature_fn(**input_dict)
    logits = output_dict["coupling_constants"]
    logits = np.round(logits.numpy(), 2)
    couplings = np.empty(shape=(0,4), dtype=str)
    for predictions in logits:
        coupling = ''
        for i, value in enumerate(predictions):
            if i == 0 and (value == 0.0 or -0.0):
                coupling += '-'
                break
            if value == 0.0 or -0.0:
                continue
            coupling += str(value)
            coupling += ';'

        couplings = np.append(couplings, coupling[:-1])
        
            
    return couplings

if __name__ == "__main__":
    path = "C:/Users/niels/Documents/repo/CASCADE/"
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"

    matching_df = pd.read_csv(path + "code/matching_model/matching_data.csv.gz", index_col=0)

    mol_df, atom_df, bond_df, distance_df = create_dictionary(2, path, type=type, smiles="OC12CC1CC(=O)OC2")
    shift = predict_shift(path, create_single_tensor(mol_df, atom_df, bond_df, distance_df, type="Shift"))
    atom_df.loc[atom_df["atom_symbol"] == "H", "Shift"] = shift
    #atom_df.loc[atom_df["atom_symbol"] == "H", "Shift"] = [2.53, 2.52, 2.73, 4.01, 1.76, 2.06, 3.72, 3.95, 4.66, 4.82]
    print(atom_df.loc[atom_df["atom_symbol"] == "H", ["mol_id", "atom_idx", "Shift", "Shape", "Coupling"]])

    shape = predict_shape(path, create_single_tensor(mol_df, atom_df, bond_df, distance_df, type="Shape"))
    atom_df.loc[atom_df["atom_symbol"] == "H", "Shape"] = shape

    print(atom_df.loc[atom_df["atom_symbol"] == "H", ["mol_id", "atom_idx", "Shift", "Shape", "Coupling"]])

    coupling = predict_coupling(path, create_single_tensor(mol_df, atom_df, bond_df, distance_df, type="Coupling"))
    atom_df["Coupling"] = atom_df["Coupling"].astype(object)
    atom_df.loc[atom_df["atom_symbol"] == "H", "Coupling"] = coupling

    print(atom_df.loc[atom_df["atom_symbol"] == "H", ["mol_id", "atom_idx", "Shift", "Shape", "Coupling"]])
    print(matching_df.loc[matching_df["mol_id"] == 1])