import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import itertools
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from tensorflow_gnn import runner
import keras.backend as K
import plotly.express as px
from lenspy import DynamicPlot
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
import gzip


import sys
import os
sys.path.append('code')
from predicting_model.create_graph_tensor import create_dictionary, create_single_tensor

def check_zero(input_graph):
    i = 0
    for item in input_graph.node_sets["_readout"].__getitem__("shift"):
        if(0 in item):
            print(i)
            print(item)
        i += 1

    #print(input_graph.context.__getitem__("smiles")[28872])

def check_data_nodes(ds, graph_spec):
    dataset = ds.map(tfgnn.keras.layers.ParseSingleExample(graph_spec))
    degree_tensor = tf.zeros([0], dtype="int64")
    formal_charge_tensor = tf.zeros([0], dtype="int64")
    is_aromatic_tensor = tf.zeros([0], dtype="int64")
    no_implicit_tensor = tf.zeros([0], dtype="int64")
    num_Hs_tensor = tf.zeros([0], dtype="int64")
    valence_tensor = tf.zeros([0], dtype="int64")

    for graph in dataset:
        degree_tensor = tf.concat([degree_tensor, graph.node_sets["atom"].__getitem__("degree")], 0)
        formal_charge_tensor = tf.concat([formal_charge_tensor, graph.node_sets["atom"].__getitem__("formal_charge")], 0)
        is_aromatic_tensor = tf.concat([is_aromatic_tensor, graph.node_sets["atom"].__getitem__("is_aromatic")], 0)
        no_implicit_tensor = tf.concat([no_implicit_tensor, graph.node_sets["atom"].__getitem__("no_implicit")], 0)
        num_Hs_tensor = tf.concat([num_Hs_tensor, graph.node_sets["atom"].__getitem__("num_Hs")], 0)
        valence_tensor = tf.concat([valence_tensor, graph.node_sets["atom"].__getitem__("valence")], 0)
    
    print("_________CARDINALITY_________")
    print("degree: " + str(len(tf.unique(degree_tensor).y)) + " (" + str(tf.unique(degree_tensor).y) + ")")
    print("formal charge: " + str(len(tf.unique(formal_charge_tensor).y)) + " (" + str(tf.unique(formal_charge_tensor).y) + ")")
    print("is aromatic: " + str(len(tf.unique(is_aromatic_tensor).y)) + " (" + str(tf.unique(is_aromatic_tensor).y) + ")")
    print("no implicit: " + str(len(tf.unique(no_implicit_tensor).y)) + " (" + str(tf.unique(no_implicit_tensor).y) + ")")
    print("num Hs: " + str(len(tf.unique(num_Hs_tensor).y)) + " (" + str(tf.unique(num_Hs_tensor).y) + ")")
    print("valence: " + str(len(tf.unique(valence_tensor).y)) + " (" + str(tf.unique(valence_tensor).y) + ")")
    print("_____________________________")

def check_data_edges(ds, graph_spec):
    dataset = ds.map(tfgnn.keras.layers.ParseSingleExample(graph_spec))
    bond_type_tensor = tf.zeros([0], dtype="float32")
    distance_tensor = tf.zeros([0], dtype="float32")
    is_conjugated_tensor = tf.zeros([0], dtype="int64")
    #stereo_tensor = tf.zeros([0], dtype="float32")

    for graph in dataset:
        bond_type_tensor = tf.concat([bond_type_tensor, graph.edge_sets["bond"].__getitem__("bond_type")], 0)
        distance_tensor = tf.concat([distance_tensor, graph.edge_sets["bond"].__getitem__("distance")], 0)
        is_conjugated_tensor = tf.concat([is_conjugated_tensor, graph.edge_sets["bond"].__getitem__("is_conjugated")], 0)
        #stereo_tensor = tf.concat([stereo_tensor, graph.edge_sets["bond"].__getitem__("stereo")], 0)

    print("_________CARDINALITY_________")
    print("bond type: " + str(len(tf.unique(bond_type_tensor).y)) + " (" + str(tf.unique(bond_type_tensor).y) + ")")
    print("distance: " + str(len(tf.unique(distance_tensor).y)) + " (" + str(tf.unique(distance_tensor).y) + ")")
    print("is conjugated: " + str(len(tf.unique(is_conjugated_tensor).y)) + " (" + str(tf.unique(is_conjugated_tensor).y) + ")")
    #print("stereo: " + str(len(tf.unique(stereo_tensor).y)) + " (" + str(tf.unique(stereo_tensor).y) + ")")
    print("_____________________________")

def check_distance_matrix():
    smiles="O"
    mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(mol, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.rdMolTransforms.CanonicalizeMol(mol, normalizeCovar=True, ignoreHs=False)
    distance_matrix = Chem.Get3DDistanceMatrix(mol)
    print("_____________Normal_____________")
    print(distance_matrix)
    mol = Chem.AddHs(mol, addCoords=True)
    distance_matrix = Chem.Get3DDistanceMatrix(mol)
    print("_____________H_added_____________")
    print(distance_matrix)

    mol_with_h = Chem.MolFromSmiles(smiles)
    mol_with_h = Chem.AddHs(mol_with_h)
    AllChem.EmbedMolecule(mol_with_h, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol_with_h)
    Chem.rdMolTransforms.CanonicalizeMol(mol_with_h, normalizeCovar=True, ignoreHs=False)
    distance_matrix_h = Chem.Get3DDistanceMatrix(mol_with_h)
    print("_____________H_at_start_____________")
    print(distance_matrix_h)

    atom_data = pd.read_csv("code/predicting_model/Shift/DFTNN/own_data_atom.csv.gz", index_col=0)
    print(atom_data)
    print(atom_data["num_exp_H"])
    H_indices = atom_data.index[atom_data["atom_symbol"] == "H"].tolist()
    print(atom_data["Shift"][H_indices])

def find_DFT_mol(path, mol_id):
    with gzip.open(path + "data/DFT8K/DFT.sdf.gz", 'rb') as dft:
        mol_suppl = Chem.ForwardSDMolSupplier(dft, sanitize=False, removeHs=False)
        for mol in mol_suppl:
            if int(mol.GetProp("_Name")) == mol_id:
                return mol
            
def convert_shape_one_hot(one_hot_matrix):
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

    return shape


def evaluate_model(dataset, model):
    num_samples = 100
    output = {"mol_id":[], "molecule":[], "mae":[], "reverse_mae":[]}
    examples = next(iter(dataset.batch(num_samples)))
    for i in tqdm(range(num_samples)):
        example = tf.reshape(examples[i], (1,))
        input_graph = tfgnn.parse_example(graph_spec, example)
        input_dict = {"examples": example}
        output_dict = signature_fn(**input_dict)
        logits = output_dict["shifts"]
        labels = tf.transpose(input_graph.node_sets["_readout"].__getitem__("shift").to_tensor())
        molecule = input_graph.context.__getitem__("smiles")
        mol_id = input_graph.context.__getitem__("_mol_id")
        mae = tf.math.reduce_mean(tf.keras.losses.MeanAbsoluteError().call(y_true=labels, y_pred=logits))
        reverse_mae = tf.math.reduce_mean(tf.keras.losses.MeanAbsoluteError().call(y_true=K.reverse(labels,axes=0), y_pred=logits))
        
        output["mol_id"].append(tf.get_static_value(mol_id)[0][0])
        output["molecule"].append(tf.get_static_value(molecule).astype(str))
        output["mae"].append(tf.get_static_value(mae))
        output["reverse_mae"].append(tf.get_static_value(reverse_mae))
    
    output_df = pd.DataFrame.from_dict(output)
    print(output_df)
    #output_df.to_csv("/home1/s3665828/code/CASCADE/data/own_data/DFT_data_results.csv.gz", compression='gzip')
    #print(output_df)

def evaluate_model_shifts(dataset, model, path):
    num_samples = 90464
    output = {"molecule":[], "mol_id":[], "index":[], "target_shift":[], "predicted_shift":[], "mae":[]}
    signature_fn = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    examples = next(iter(dataset.batch(num_samples)))
    for i in tqdm(range(34117, 34120)):
        example = tf.reshape(examples[i], (1,))
        input_graph = tfgnn.parse_example(graph_spec, example)
        input_dict = {"examples": example}
        output_dict = signature_fn(**input_dict)
        logits = tf.squeeze(output_dict["shifts"])
        labels = tf.squeeze(tf.transpose(input_graph.node_sets["_readout"].__getitem__("shift").to_tensor()))
        smiles = input_graph.context.__getitem__("smiles")
        smiles = tf.get_static_value(tf.squeeze(smiles))
        print(smiles)
        mol_id = input_graph.context.__getitem__("_mol_id")
        mol_id = tf.get_static_value(tf.squeeze(mol_id))
        index = input_graph.edge_sets["_readout/shift"].adjacency.source[0]
        if logits.ndim == 0:
            logits = [logits]
            labels = [labels]
        
        for j, predicted_shift in enumerate(logits):
            target_shift = labels[j]
            output["molecule"].append(smiles)
            output["mol_id"].append(mol_id)
            output["index"].append(tf.get_static_value(index[j]))
            output["target_shift"].append(tf.get_static_value(target_shift))
            output["predicted_shift"].append(tf.get_static_value(predicted_shift))
            mae = tf.keras.losses.MeanAbsoluteError().call(y_true=tf.expand_dims(target_shift, axis=-1), y_pred=tf.expand_dims(predicted_shift, axis=-1))
            output["mae"].append(tf.get_static_value(mae))
            
    print("done")
    #output_df = pd.DataFrame.from_dict(output)
    #print(output_df)
    #output_df.to_csv(path + "data/own_data/DFT_shift_results.csv.gz", compression='gzip')


def evaluate_model_shapes(dataset, model, path):
    num_samples = 1000
    output = {"molecule":[], "mol_id":[], "index":[], "target_shape":[], "predicted_shape":[], "cce":[], 
              "weighted_cce":[], "pred_1":[], "pred_2":[], "pred_3":[], "pred_4":[],
              "target_1":[], "target_2":[], "target_3":[], "target_4":[]}
    signature_fn = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    examples = next(iter(dataset.batch(num_samples)))
    for i in tqdm(range(num_samples)):
        example = tf.reshape(examples[i], (1,))
        input_graph = tfgnn.parse_example(graph_spec, example)
        input_dict = {"examples": example}
        output_dict = signature_fn(**input_dict)
        logits = output_dict["shape"]
        labels = input_graph.node_sets["_readout"].__getitem__("shape").to_tensor()[0]
        smiles = input_graph.context.__getitem__("smiles")
        smiles = tf.get_static_value(smiles).astype(str)[0][0]
        mol_id = input_graph.context.__getitem__("_mol_id")
        mol_id = tf.get_static_value(mol_id).astype(str)[0][0]
        index = input_graph.edge_sets["_readout/shape"].adjacency.source[0]
        

        for j, predicted_shape in enumerate(logits):
            converted_labels = convert_shape_one_hot(labels[j])
            #print(predicted_shape)
            converted_predictions = convert_shape_one_hot(predicted_shape)
            target_shape = labels[j]
            #print(target_shape)
            output["molecule"].append(smiles)
            output["mol_id"].append(mol_id)
            output["index"].append(tf.get_static_value(index[j]))
            cce = tf.keras.losses.CategoricalCrossentropy()
            output["cce"].append(tf.get_static_value(tf.math.reduce_mean(cce(y_true=target_shape, y_pred=predicted_shape))))
            output["weighted_cce"].append(tf.get_static_value(tf.math.reduce_mean(cce(y_true=target_shape, y_pred=predicted_shape, sample_weight=[1.0,0.4,0.15,0.05]))))
            output["target_shape"].append(tf.get_static_value(converted_labels))
            output["predicted_shape"].append(tf.get_static_value(converted_predictions))

            output["pred_1"].append(converted_predictions[0])
            output["pred_2"].append(converted_predictions[1])
            output["pred_3"].append(converted_predictions[2])
            output["pred_4"].append(converted_predictions[3])

            output["target_1"].append(converted_labels[0])
            output["target_2"].append(converted_labels[1])
            output["target_3"].append(converted_labels[2])
            output["target_4"].append(converted_labels[3])

    
    output_df = pd.DataFrame.from_dict(output)
    print(output_df)
    total = len(output_df)
    print("Correct prediction: " + str(round((len(output_df.loc[output_df["predicted_shape"] == output_df["target_shape"]])/total)*100, 2)) + "%")
    print("First token correct: " + str(round((len(output_df.loc[output_df["pred_1"] == output_df["target_1"]])/total)*100, 2)) + "%")
    print("Second token correct: " + str(round((len(output_df.loc[output_df["pred_2"] == output_df["target_2"]])/total)*100, 2)) + "%")
    print("Third token correct: " + str(round((len(output_df.loc[output_df["pred_3"] == output_df["target_3"]])/total)*100, 2)) + "%")
    print("Fourth token correct: " + str(round((len(output_df.loc[output_df["pred_4"] == output_df["target_4"]])/total)*100, 2)) + "%")
    print("Mean CCE: " + str(round(output_df["cce"].mean(),2)))
    print("Mean Weighted CCE: " + str(round(output_df["weighted_cce"].mean(),2)))
    print(output_df["pred_1"].value_counts(normalize=True) * 100)
    print(output_df["pred_2"].value_counts(normalize=True) * 100)
    print(output_df["pred_3"].value_counts(normalize=True) * 100)
    print(output_df["pred_4"].value_counts(normalize=True) * 100)
    output_df.to_csv(path + "data/own_data/shape_results.csv.gz", compression='gzip')


def evaluate_model_coupling(dataset, model, path):
    num_samples = 63324
    output = {"molecule":[], "mol_id":[], "index":[], "target_coupling":[], "predicted_coupling":[],
              "mae":[], "pred_1":[], "pred_2":[], "pred_3":[], "pred_4":[],
              "target_1":[], "target_2":[], "target_3":[], "target_4":[],}
    signature_fn = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    examples = next(iter(dataset.batch(num_samples)))
    for i in tqdm(range(34116, num_samples)):
        example = tf.reshape(examples[i], (1,))
        input_graph = tfgnn.parse_example(graph_spec, example)
        input_dict = {"examples": example}
        output_dict = signature_fn(**input_dict)
        logits = np.round(output_dict["coupling_constants"], 2)
        labels = input_graph.node_sets["_readout"].__getitem__("coupling").to_tensor()[0]
        smiles = input_graph.context.__getitem__("smiles")
        smiles = tf.get_static_value(smiles).astype(str)[0][0][0]
        mol_id = input_graph.context.__getitem__("_mol_id")
        mol_id = tf.get_static_value(mol_id).astype(str)[0][0][0]
        index = input_graph.edge_sets["_readout/coupling"].adjacency.source[0]
        if logits.ndim == 1:
            logits = [logits]

        for j, predicted_couplings in enumerate(logits):
            print(predicted_couplings)
            target_couplings = labels[j]
            output["molecule"].append(smiles)
            output["mol_id"].append(mol_id)
            output["index"].append(tf.get_static_value(index[j]))
            mae = tf.keras.losses.MeanAbsoluteError(reduction="sum")
            output["mae"].append(tf.get_static_value(tf.math.reduce_mean(mae(y_true=target_couplings, y_pred=predicted_couplings))))
            output["target_coupling"].append(tf.get_static_value(target_couplings))
            output["predicted_coupling"].append(tf.get_static_value(predicted_couplings))

            output["pred_1"].append(tf.get_static_value(predicted_couplings[0]))
            output["pred_2"].append(tf.get_static_value(predicted_couplings[1]))
            output["pred_3"].append(tf.get_static_value(predicted_couplings[2]))
            output["pred_4"].append(tf.get_static_value(predicted_couplings[3]))

            output["target_1"].append(tf.get_static_value(target_couplings[0]))
            output["target_2"].append(tf.get_static_value(target_couplings[1]))
            output["target_3"].append(tf.get_static_value(target_couplings[2]))
            output["target_4"].append(tf.get_static_value(target_couplings[3]))
    
    output_df = pd.DataFrame.from_dict(output)
    print(output_df)
    print("First coupling MAE: " + str(tf.get_static_value(tf.math.reduce_mean(mae(y_true=output["pred_1"], y_pred=output["target_1"])))))
    print("Second coupling MAE: " + str(tf.get_static_value(tf.math.reduce_mean(mae(y_true=output["pred_2"], y_pred=output["target_2"])))))
    print("Third coupling MAE: " + str(tf.get_static_value(tf.math.reduce_mean(mae(y_true=output["pred_3"], y_pred=output["target_3"])))))
    print("Fourth coupling MAE: " + str(tf.get_static_value(tf.math.reduce_mean(mae(y_true=output["pred_4"], y_pred=output["target_4"])))))
    print("Mean MAE: " + str(output_df["mae"].mean()))

    output_df.to_csv(path + "data/own_data/coupling_results.csv.gz", compression='gzip')


def evaluate_model_shape_and_coupling(dataset, model, path, num):
    num_samples = 90464
    output = {"molecule":[], "mol_id":[], "index":[], "target_coupling":[], "predicted_coupling":[],
              "mae":[], "coupling_pred_1":[], "coupling_pred_2":[], "coupling_pred_3":[], "coupling_pred_4":[],
              "coupling_target_1":[], "coupling_target_2":[], "coupling_target_3":[], "coupling_target_4":[],
              "target_shape":[], "predicted_shape":[], "converted_shape":[], "cce":[], "weighted_cce":[], 
              "shape_pred_1":[], "shape_pred_2":[], "shape_pred_3":[], "shape_pred_4":[], "shape_target_1":[], 
              "shape_target_2":[], "shape_target_3":[], "shape_target_4":[]}
    signature_fn = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    examples = next(iter(dataset.batch(num_samples)))
    for i in tqdm(range(num_samples)):
        if i == 100: break
        example = tf.reshape(examples[i], (1,))
        input_graph = tfgnn.parse_example(graph_spec, example)
        input_dict = {"examples": example}
        output_dict = signature_fn(**input_dict)
        coupling_logits = np.round(output_dict["coupling_constants"], 2)
        shape_logits = output_dict["shape"]
        coupling_labels = input_graph.node_sets["_readout"].__getitem__("coupling").to_tensor()[0]
        shape_labels = input_graph.node_sets["_readout"].__getitem__("shape").to_tensor()[0]
        smiles = input_graph.context.__getitem__("smiles")
        smiles = tf.get_static_value(smiles).astype(str)[0][0][0]
        mol_id = input_graph.context.__getitem__("_mol_id")
        mol_id = tf.get_static_value(mol_id).astype(str)[0][0][0]
        index = input_graph.edge_sets["_readout/hydrogen"].adjacency.source[0]
        if coupling_logits.ndim == 1:
            coupling_logits = [coupling_logits]
            shape_logits = [shape_logits]

        for j, predicted_couplings in enumerate(coupling_logits):
            #print(predicted_couplings)
            target_couplings = coupling_labels[j]
            target_couplings = tf.get_static_value(target_couplings).flatten()
            predicted_couplings = tf.get_static_value(predicted_couplings).flatten()

            output["molecule"].append(smiles)
            output["mol_id"].append(mol_id)
            output["index"].append(tf.get_static_value(index[j]))
            mae = tf.keras.losses.MeanAbsoluteError(reduction="sum")
            output["mae"].append(tf.get_static_value(tf.math.reduce_mean(mae(y_true=target_couplings, y_pred=predicted_couplings))))
            output["target_coupling"].append(target_couplings)
            output["predicted_coupling"].append(predicted_couplings)

            output["coupling_pred_1"].append(tf.get_static_value(predicted_couplings[0]))
            output["coupling_pred_2"].append(tf.get_static_value(predicted_couplings[1]))
            output["coupling_pred_3"].append(tf.get_static_value(predicted_couplings[2]))
            output["coupling_pred_4"].append(tf.get_static_value(predicted_couplings[3]))

            output["coupling_target_1"].append(tf.get_static_value(target_couplings[0]))
            output["coupling_target_2"].append(tf.get_static_value(target_couplings[1]))
            output["coupling_target_3"].append(tf.get_static_value(target_couplings[2]))
            output["coupling_target_4"].append(tf.get_static_value(target_couplings[3]))

            predicted_shape = shape_logits[j]
            converted_labels = convert_shape_one_hot(shape_labels[j])
            converted_predictions = convert_shape_one_hot(predicted_shape)
            target_shape = shape_labels[j]

            cce = tf.keras.losses.CategoricalCrossentropy()
            output["cce"].append(tf.get_static_value(tf.math.reduce_mean(cce(y_true=target_shape, y_pred=predicted_shape))))
            output["weighted_cce"].append(tf.get_static_value(tf.math.reduce_mean(cce(y_true=target_shape, y_pred=predicted_shape, sample_weight=[1.0,0.4,0.15,0.05]))))
            output["target_shape"].append(tf.get_static_value(converted_labels))
            output["predicted_shape"].append(tf.get_static_value(predicted_shape))
            output["converted_shape"].append(tf.get_static_value(converted_predictions))

            output["shape_pred_1"].append(converted_predictions[0])
            output["shape_pred_2"].append(converted_predictions[1])
            output["shape_pred_3"].append(converted_predictions[2])
            output["shape_pred_4"].append(converted_predictions[3])

            output["shape_target_1"].append(converted_labels[0])
            output["shape_target_2"].append(converted_labels[1])
            output["shape_target_3"].append(converted_labels[2])
            output["shape_target_4"].append(converted_labels[3])

    
    output_df = pd.DataFrame.from_dict(output)
    print(output_df[["molecule", "index", "target_shape", "target_coupling", "predicted_shape", "predicted_coupling", "cce", "mae"]].to_string())
    total = len(output_df)  
    print("First coupling MAE: " + str(tf.get_static_value(tf.math.reduce_mean(mae(y_true=output["coupling_pred_1"], y_pred=output["coupling_target_1"])))))
    print("Second coupling MAE: " + str(tf.get_static_value(tf.math.reduce_mean(mae(y_true=output["coupling_pred_2"], y_pred=output["coupling_target_2"])))))
    print("Third coupling MAE: " + str(tf.get_static_value(tf.math.reduce_mean(mae(y_true=output["coupling_pred_3"], y_pred=output["coupling_target_3"])))))
    print("Fourth coupling MAE: " + str(tf.get_static_value(tf.math.reduce_mean(mae(y_true=output["coupling_pred_4"], y_pred=output["coupling_target_4"])))))
    print("Mean MAE: " + str(output_df["mae"].mean()))

    print("Correct prediction: " + str(round((len(output_df.loc[output_df["converted_shape"] == output_df["target_shape"]])/total)*100, 2)) + "%")
    print("First token correct: " + str(round((len(output_df.loc[output_df["shape_pred_1"] == output_df["shape_target_1"]])/total)*100, 2)) + "%")
    print("Second token correct: " + str(round((len(output_df.loc[output_df["shape_pred_2"] == output_df["shape_target_2"]])/total)*100, 2)) + "%")
    print("Third token correct: " + str(round((len(output_df.loc[output_df["shape_pred_3"] == output_df["shape_target_3"]])/total)*100, 2)) + "%")
    print("Fourth token correct: " + str(round((len(output_df.loc[output_df["shape_pred_4"] == output_df["shape_target_4"]])/total)*100, 2)) + "%")
    print("Mean CCE: " + str(round(output_df["cce"].mean(),2)))
    print("Mean Weighted CCE: " + str(round(output_df["weighted_cce"].mean(),2)))
    print(output_df["shape_pred_1"].value_counts(normalize=True) * 100)
    print(output_df["shape_pred_2"].value_counts(normalize=True) * 100)
    print(output_df["shape_pred_3"].value_counts(normalize=True) * 100)
    print(output_df["shape_pred_4"].value_counts(normalize=True) * 100)

    output_df.to_csv(path + "data/own_data/shape_and_coupling_results" + str(num) + ".csv.gz", compression='gzip')



def predict_shift(path, data):
    graph_schema = tfgnn.read_schema(path + "code/predicting_model/GraphSchema.pbtxt")
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    shift_model = tf.saved_model.load(path + "code/predicting_model/Shift/gnn/models/shift_model_best")
    signature_fn = shift_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    example = tf.reshape(data, (1,))
    input_graph = tfgnn.parse_example(graph_spec, example)
    input_dict = {"examples": example}
    output_dict = signature_fn(**input_dict)
    logits = output_dict["shifts"]

    return logits.numpy().flatten()

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
    shapes = [convert_shape_one_hot(x) for x in logits]
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
        
    couplings = ["-" if coupling == '' else coupling for coupling in couplings]
    return couplings

def predict_all(path, data):
    result = {"molecule":[], "index":[], "shift":[], "shape":[], "couplings":[]}
    for smiles in data:
        mol_df, atom_df, bond_df, distance_df = create_dictionary(2, path, type=type, smiles=smiles)
        shift = predict_shift(path, create_single_tensor(mol_df, atom_df, bond_df, distance_df, type="Shift"))
        shape = predict_shape(path, create_single_tensor(mol_df, atom_df, bond_df, distance_df, type="Shape"))
        coupling = predict_coupling(path, create_single_tensor(mol_df, atom_df, bond_df, distance_df, type="Coupling"))
        #atom_df["Coupling"] = atom_df["Coupling"].astype(object)
        result["molecule"].extend([mol_df["smiles"].values[0]] * mol_df["n_pro"].values[0])
        result["index"].extend(atom_df.loc[atom_df["atom_symbol"] == "H", "atom_idx"].values)
        result["shift"].extend(shift)
        result["shape"].extend(shape)
        result["couplings"].extend(coupling)
        
    results = pd.DataFrame.from_dict(result)
    results.to_csv(path + "data/own_data/all_results.csv.gz", compression='gzip')
    print(results)
    return results


def check_sample(dataset):
    examples = next(iter(dataset.batch(63565)))
    sample = tf.reshape(examples[832], (1,))
    graph = tfgnn.parse_example(graph_spec, sample)
    print(graph.node_sets["atom"].__getitem__("atom_num"))
    print(graph.edge_sets["bond"].adjacency.source)
    print(graph.edge_sets["bond"].adjacency.target)

    smiles = "O=CNC=NC1CC1O"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol, addCoords=True)
    for n, atom in enumerate(mol.GetAtoms()):
        print(atom.GetSymbol())
        for bond in atom.GetBonds():
            print(bond.GetBeginAtomIdx())
            print(bond.GetEndAtomIdx())
            

def evaluate_sample(dataset, model, index):
    output = {"molecule":[], "mae":[], "reverse_mae":[]}
    examples = next(iter(dataset.batch(90780)))
    example = tf.reshape(examples[index], (1,))
    input_graph = tfgnn.parse_example(graph_spec, example)
    input_dict = {"examples": example}
    output_dict = signature_fn(**input_dict)
    logits = output_dict["shifts"]
    labels = tf.transpose(input_graph.node_sets["_readout"].__getitem__("shift").to_tensor())
    molecule = input_graph.context.__getitem__("smiles")
    mae = tf.math.reduce_mean(tf.keras.losses.MeanAbsoluteError().call(y_true=labels, y_pred=logits))
    reverse_mae = tf.math.reduce_mean(tf.keras.losses.MeanAbsoluteError().call(y_true=K.reverse(labels,axes=0), y_pred=logits))
    
    output["molecule"].append(tf.get_static_value(molecule).astype(str)[0][0])
    output["mae"].append(tf.get_static_value(mae))
    output["reverse_mae"].append(tf.get_static_value(reverse_mae))

    output_df = pd.DataFrame.from_dict(output)
    print(output["molecule"])
    print("Molecule labels")
    print(labels)
    print("Model shifts")
    print(logits)
    print(output_df)

def evaluate_molecule(model, smiles):
    output = {"molecule":[], "index":[], "predicted_shift":[]}
    #example = process_samples(2, "", smiles="C#CCC1CCOCO1")
    example = tf.reshape(example, (1,))
    input_graph = tfgnn.parse_example(graph_spec, example)
    input_dict = {"examples": example}
    output_dict = signature_fn(**input_dict)
    logits = output_dict["shifts"]

    molecule = input_graph.context.__getitem__("smiles")

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "H":
            output["index"].append(atom.GetIdx())
    
    for shift in logits:
        output["molecule"].append(tf.get_static_value(molecule).astype(str)[0][0])
        output["predicted_shift"].append(tf.get_static_value(shift)[0])

    output_df = pd.DataFrame.from_dict(output)
    print(output_df)
    return output_df

def plot_results_scatter(results):
    fig = px.scatter(results, x=results.index, y='mae')
    '''fig.add_trace(go.Scatter(x=results.index, y=results['mae'], mode='markers', name='mae'))
    fig.add_trace(go.Scatter(x=results.index, y=results['reverse_mae'], mode='markers', name='reverse_mae'))'''
    fig.update_layout(xaxis=dict(type='category'))
    plot = DynamicPlot(fig)
    plot.show()

def plot_results_hist(results):
    fig = px.histogram(results, x='mae', nbins=100)
    plot = DynamicPlot(fig)
    plot.show()

def plot_shift_errors(shift_results):
    fig = px.histogram(shift_results, x='target_shift', y='mae', nbins=50, histfunc='avg')
    plot = DynamicPlot(fig)
    plot.show()

    results = results.sort_values(by=['mae'], ascending=False)
    print(results[:100])

def plot_shifts(path):
    atom_data = pd.read_csv(path + "code/predicting_model/Shift/DFTNN/own_data_atom.csv.gz")
    print(atom_data.loc[atom_data["atom_symbol"] == "H"]["Shift"])
    fig = px.histogram(atom_data.loc[atom_data["atom_symbol"] == "H"], x='Shift', nbins=50)
    plot = DynamicPlot(fig)
    plot.show()

def print_stats(path):
    results = pd.read_csv(path + "data/own_data/DFT_results.csv.gz", index_col=0)
    results["reverse_better"] =  np.where(results['reverse_mae'] < results["mae"], "yes", "no")
    print("Average MAE:\n" + str(results.loc[:, "mae"].mean()))
    print("Number of samples where reverse is better: " + str(results["reverse_better"].value_counts()[1]))
    print("Samples where reverse is better: " + str(results[results["reverse_better"]=="yes"]))
    print("Number of samples where MAE > 1: " + str(len(results[results["mae"]>1])))
    print("Samples where MAE > 1:\n" + str(results[results["mae"]>1]))

def visualize_shifts(path, results):
    smiles = results["molecule"][0]
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    d = Draw.rdMolDraw2D.MolDraw2DCairo(500, 500)
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        matching_data = results.loc[results["index"] == index]
        if not matching_data.empty:
            rounded_shift = round(matching_data["predicted_shift"].values[0], 2)
            atom.SetProp('atomNote', str(rounded_shift))
    
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    d.WriteDrawingText(path + "data/own_data/mol.png")

    mol_image = plt.imread(path + "data/own_data/mol.png")
    fig = plt.figure(figsize=(10,8))
    fig = plt.imshow(mol_image)
    plt.axis('off')
    plt.show()
    
def visualize_errors(path, results, dft=False):
    i = 0
    with PdfPages(path + 'data/own_data/output.pdf') as pdf:
        for smiles in results["molecule"].unique():
            if i == 100: break
            sample_results = results.loc[results["molecule"] == smiles]

            if dft:
                mol = find_DFT_mol(path, sample_results["mol_id"].iloc[0])
            else:
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)

            AllChem.Compute2DCoords(mol)
            d = Draw.rdMolDraw2D.MolDraw2DCairo(500, 500)
            atom_data = results.loc[results["molecule"] == smiles]
            highlightAtoms = []
                        #  red             yellow           green
            colours=[(1.0, 0.0, 0.0),(1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
            atom_cols = {}

            for atom in mol.GetAtoms():
                index = atom.GetIdx()
                print(index)
                matching_data = atom_data.loc[atom_data["index"] == index]
                if not matching_data.empty:
                    mae = matching_data["mae"].values[0]
                    rounded_shift = round(matching_data["predicted_shift"].values[0], 2)
                    atom.SetProp('atomNote', str(rounded_shift))
                    highlightAtoms.append(index)
                    if mae <= 1.0:   # interpolate between green and yellow
                        yellow = colours[1]
                        green = colours[2]
                        colour = (green[0] + (yellow[0] * mae), green[1], green[2])
                        atom_cols[index] = colour
                    else:           # interpolate between yellow and red
                        yellow = colours[1]
                        red = colours[0]
                        colour = (red[0], max(red[1] + (yellow[1] - (1.0 * ((mae-1.0)/9.0))), 0), red[2])
                        atom_cols[index] = colour

            Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=highlightAtoms, highlightAtomColors=atom_cols)
            d.FinishDrawing()
            d.WriteDrawingText(path + "data/own_data/mol.png")

            mol_image = plt.imread(path + "data/own_data/mol.png")
            colormap = plt.imread(path + "data/own_data/colormap.png")

            fig = plt.figure(figsize=(10,8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[8, 1])
            ax0 = plt.subplot(gs[0])
            ax0.set_title(smiles)
            ax0.text(210, 17, "Mol ID: " + str(sample_results["mol_id"].values[0]))
            ax0.text(210, 7, "Mean MAE: " + str(round(sample_results["mae"].mean(), 2)))
            print(smiles)
            print(sample_results["mol_id"].iloc[0])
            print(sample_results.drop(["molecule", "mol_id"], axis=1))
            print("Mean MAE: " + str(round(sample_results["mae"].mean(), 2)))
            ax0.imshow(mol_image)
            plt.axis('off')
            ax1 = plt.subplot(gs[1])
            ax1.imshow(colormap)
            plt.axis('off')

            plt.tight_layout()
            #plt.show()

            pdf.savefig(fig, bbox_inches='tight')

            i += 1


    '''x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    colours = ((0.0, colours[2]), (0.1, colours[1]), (1.0, colours[0]))
    
    cmap = LinearSegmentedColormap.from_list("MAE", colours, 1000)
    plt.scatter(x=x, y=y, c=x, cmap=cmap)
    plt.colorbar(label="MAE", orientation="horizontal")
    plt.show()'''

def visualize_all(path, results):
    with PdfPages(path + 'data/own_data/output.pdf') as pdf:
        for smiles in results["molecule"].unique():
            sample_results = results.loc[results["molecule"] == smiles]

            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)

            AllChem.Compute2DCoords(mol)
            d = Draw.rdMolDraw2D.MolDraw2DCairo(500, 500)
            atom_data = results.loc[results["molecule"] == smiles]

            for atom in mol.GetAtoms():
                index = atom.GetIdx()
                matching_data = atom_data.loc[atom_data["index"] == index]
                if not matching_data.empty:
                    rounded_shift = round(matching_data["shift"].values[0], 2)
                    atom.SetProp('atomNote', str(rounded_shift) + "<sup>" + 
                                 str(matching_data["shape"].values[0]) + "</sup><sub>" +
                                 str(matching_data["couplings"].values[0]) + "</sub>")
                    
            Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
            d.FinishDrawing()
            d.WriteDrawingText(path + "data/own_data/mol.png")

            mol_image = plt.imread(path + "data/own_data/mol.png")

            fig = plt.figure(figsize=(10,8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[8, 1])
            ax0 = plt.subplot(gs[0])
            ax0.set_title(smiles)
            print(smiles)
            print(sample_results.drop(["molecule"], axis=1))
            ax0.imshow(mol_image)
            plt.axis('off')

            plt.tight_layout()
            #plt.show()

            pdf.savefig(fig, bbox_inches='tight')


def check_graph(dataset):
    num_samples = 10
    for i in range(num_samples):
        examples = next(iter(dataset.batch(num_samples)))
        example = tf.reshape(examples[i], (1,))
        input_graph = tfgnn.parse_example(graph_spec, example)
        print(input_graph)
        print(input_graph.node_sets["_readout"].__getitem__("coupling"))

        
def count_invalid_shape(predictions):    # A shape is invalid if the any token except the first is "m" or if any token after an "s" is not "s"
    invalid_count_0 = 0
    invalid_count_1 = 0
    invalid_count_2 = 0
    for shape in predictions["converted_shape"]:
        first_s = shape.find('s')
        if ('m' in shape[1:]): # if another token than the first is m
            invalid_count_0 += 1
        elif (first_s != -1) and any(char != 's' for char in shape[first_s:]): # if there's a non-s after an s
            invalid_count_1 += 1
        elif (shape[0] == 'm') and any(char != 's' for char in shape[1:]): # if there's a non-s after an m
            invalid_count_2 += 1

    invalid_count = invalid_count_0 + invalid_count_1 + invalid_count_2
    print("m in wrong place: " + str(invalid_count_0))
    print("non-s after s: " + str(invalid_count_1))
    print("non-s after m: " + str(invalid_count_2))
    print("Total invalid shapes: " + str(invalid_count))
    return invalid_count


def count_invalid_couplings(predictions):    # A coupling is invalid if any value after a 0.0 is non-zero or if a token with an s or m as shape has a value for the coupling constant
    invalid_count_0 = 0
    invalid_count_1 = 0
    invalid_count_2 = 0
    for index, prediction in tqdm(predictions.iterrows()):
        couplings = [prediction["coupling_pred_1"], prediction["coupling_pred_2"], prediction["coupling_pred_3"], prediction["coupling_pred_4"]]
        shape = prediction["converted_shape"]
        if (couplings[0] == 0.0) and any(couplings[1:]):  # non-zero coupling after zero coupling
            invalid_count_0 += 1
        elif (couplings[1] == 0.0) and any(couplings[2:]):
            invalid_count_0 += 1
        elif (couplings[2] == 0.0) and any(couplings[3:]):
            invalid_count_0 += 1
        elif ((shape == "msss") or (shape == "ssss")) and (any(couplings)): # if there are coupling constants for an m or s shape
            invalid_count_1 += 1
        elif sum(1 for x in couplings if x != 0.0) != sum(1 for s in shape if (s != 'm' and s != 's')): # if the number of coupling constants doesn't align with the number of non-s or non-m shapes
            invalid_count_2 += 1

    invalid_count = invalid_count_0 + invalid_count_1 + invalid_count_2
    print("non-zero after 0: " + str(invalid_count_0))
    print("couplings for m or s: " + str(invalid_count_1))
    print("different number of couplings and shapes: " + str(invalid_count_2))
    print("Total invalid couplings: " + str(invalid_count))
    return invalid_count


if __name__ == "__main__":
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    #path = "/home1/s3665828/code/CASCADE/"
    path = "C:/Users/niels/Documents/repo/CASCADE/"

    print("Model 0:")
    count_invalid_shape(pd.read_csv(path + "data/own_data/shape_and_coupling_results0.csv.gz"))
    count_invalid_couplings(pd.read_csv(path + "data/own_data/shape_and_coupling_results0.csv.gz"))
    print("_______________________")
    print("Model 1:")
    count_invalid_shape(pd.read_csv(path + "data/own_data/shape_and_coupling_results1.csv.gz"))
    count_invalid_couplings(pd.read_csv(path + "data/own_data/shape_and_coupling_results1.csv.gz"))
    print("_______________________")
    print("Model 2:")
    count_invalid_shape(pd.read_csv(path + "data/own_data/shape_and_coupling_results2.csv.gz"))
    count_invalid_couplings(pd.read_csv(path + "data/own_data/shape_and_coupling_results2.csv.gz"))