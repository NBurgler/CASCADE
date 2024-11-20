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
from predicting_model.create_graph_tensor import process_samples

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
    for i in tqdm(range(num_samples)):
        example = tf.reshape(examples[i], (1,))
        continue
        input_graph = tfgnn.parse_example(graph_spec, example)
        input_dict = {"examples": example}
        output_dict = signature_fn(**input_dict)
        logits = output_dict["shifts"]
        labels = tf.transpose(input_graph.node_sets["_readout"].__getitem__("shift").to_tensor())
        smiles = input_graph.context.__getitem__("smiles")
        smiles = tf.get_static_value(smiles).astype(str)[0][0]
        mol_id = input_graph.context.__getitem__("_mol_id")
        mol_id = tf.get_static_value(mol_id).astype(str)[0][0]
        index = input_graph.edge_sets["_readout/shift"].adjacency.source[0]
        
        for j, predicted_shift in enumerate(logits):
            target_shift = labels[j]
            output["molecule"].append(smiles)
            output["mol_id"].append(mol_id)
            output["index"].append(tf.get_static_value(index[j]))
            output["target_shift"].append(tf.get_static_value(target_shift)[0])
            output["predicted_shift"].append(tf.get_static_value(predicted_shift)[0])
            mae = tf.math.reduce_mean(tf.keras.losses.MeanAbsoluteError().call(y_true=target_shift, y_pred=predicted_shift))
            output["mae"].append(tf.get_static_value(mae))
            
    print("done")
    #output_df = pd.DataFrame.from_dict(output)
    #print(output_df)
    #output_df.to_csv(path + "data/own_data/DFT_shift_results.csv.gz", compression='gzip')


def evaluate_model_shapes(dataset, model, path):
    num_samples = 63324
    #output = {"molecule":[], "mol_id":[], "index":[], "target_shape":[], "predicted_shape":[], "cce":[], 
    #          "pred_1":[], "pred_2":[], "pred_3":[], "pred_4":[], "pred_5":[], "pred_6":[],
    #          "target_1":[], "target_2":[], "target_3":[], "target_4":[], "target_5":[], "target_6":[],}
    output = {"molecule":[], "mol_id":[], "index":[], "target_shape":[], "predicted_shape":[], "cce":[], 
              "weighted_cce":[], "pred_1":[], "pred_2":[], "pred_3":[], "pred_4":[],
              "target_1":[], "target_2":[], "target_3":[], "target_4":[],}
    signature_fn = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    examples = next(iter(dataset.batch(num_samples)))
    for i in tqdm(range(0, 10)):
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
            #output["pred_5"].append(converted_predictions[4])
            #output["pred_6"].append(converted_predictions[5])

            output["target_1"].append(converted_labels[0])
            output["target_2"].append(converted_labels[1])
            output["target_3"].append(converted_labels[2])
            output["target_4"].append(converted_labels[3])
            #output["target_5"].append(converted_labels[4])
            #output["target_6"].append(converted_labels[5])

    
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
    output_df.to_csv(path + "data/own_data/shape_results.csv.gz", compression='gzip')


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
    example = process_samples(2, "", smiles="C#CCC1CCOCO1")
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

def check_graph(dataset):
    num_samples = 7454
    for i in range(5287, 5317):
        examples = next(iter(dataset.batch(num_samples)))
        example = tf.reshape(examples[i], (1,))
        input_graph = tfgnn.parse_example(graph_spec, example)
        mol_id = input_graph.context.__getitem__("_mol_id")
        mol_id = tf.get_static_value(mol_id).astype(str)[0][0]
        if mol_id == str(20045567):
            print(input_graph.context.__getitem__("smiles")[0])
            print(input_graph.node_sets["atom"].__getitem__("chiral_tag")[0])
            return input_graph.node_sets["atom"].__getitem__("chiral_tag")[0]
        


if __name__ == "__main__":
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    #path = "/home1/s3665828/code/CASCADE/"
    path = "C:/Users/niels/Documents/repo/CASCADE/"

    graph_schema = tfgnn.read_schema(path + "code/predicting_model/GraphSchemaMult.pbtxt")
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    #model = tf.saved_model.load(path + "code/predicting_model/Shift/DFTNN/gnn/models/DFT_model_new")
    model = tf.saved_model.load(path + "code/predicting_model/Shape/gnn/models/mult_weight_model_4")
    signature_fn = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    
    dataset_provider = runner.TFRecordDatasetProvider(filenames=[path + "data/own_data/Shape/own_4_train.tfrecords"])
    dataset = dataset_provider.get_dataset(tf.distribute.InputContext())

    evaluate_model_shapes(dataset, model, path)

    #plot_results_hist(pd.read_csv(path + "data/own_data/own_data_results.csv.gz", index_col=0))
    #plot_shift_errors(pd.read_csv(path + "data/own_data/own_shift_results.csv.gz", index_col=0))

    #evaluate_model_shifts(dataset, model, path)
    #evaluate_model(dataset, model)
    #evaluate_sample(dataset, model, 2)
    #plot_shift_errors(path)

    #visualize_shifts(path, evaluate_molecule(model, "C#CCC1CCOCO1"))
    #data = pd.read_csv(path + "data/own_data/own_shift_results.csv.gz", index_col=0)
    #data = data.sort_values(by="mae", ascending=False)
    #visualize_errors(path, data, dft=False)
    '''
    mean_mae_list = {"mol_id":[], "mean_mae":[]}
    for mol_id in tqdm(data["mol_id"].unique()):
        atom_data = data.loc[data["mol_id"] == mol_id]
        mean_mae = str(round(atom_data["mae"].mean(), 2))
        mean_mae_list["mol_id"].append(mol_id)
        mean_mae_list["mean_mae"].append(mean_mae)

    mean_mae_df = pd.DataFrame(mean_mae_list)

    print(mean_mae_df)
    print(data)
    data = pd.merge(data, mean_mae_df, on="mol_id", how="left")

    print(data)
    data = data.sort_values(by="mean_mae", ascending=False)
    print(data)
    visualize_errors(path, data, dft=False)
    '''