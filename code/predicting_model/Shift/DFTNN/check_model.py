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

def evaluate_model(dataset, model):
    num_samples = 10
    output = {"mol_id":[], "molecule":[], "mae":[], "reverse_mae":[]}
    for i in range(10):
        examples = next(iter(dataset.batch(num_samples)))
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
    #output_df.to_csv("/home1/s3665828/code/CASCADE/data/own_data/data_results.csv.gz", compression='gzip')
    #print(output_df)

def evaluate_model_shifts(dataset, model, path):
    num_samples = 10
    output = {"molecule":[], "index":[], "target_shift":[], "predicted_shift":[], "mae":[]}
    signature_fn = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    for i in range(num_samples):
        examples = next(iter(dataset.batch(num_samples)))
        example = tf.reshape(examples[i], (1,))
        input_graph = tfgnn.parse_example(graph_spec, example)
        input_dict = {"examples": example}
        output_dict = signature_fn(**input_dict)
        logits = output_dict["shifts"]
        labels = tf.transpose(input_graph.node_sets["_readout"].__getitem__("shift").to_tensor())
        smiles = input_graph.context.__getitem__("smiles")
        smiles = tf.get_static_value(smiles).astype(str)[0][0]
        index = input_graph.node_sets["atom"].__getitem__("_atom_idx")

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "H":
                output["index"].append(atom.GetIdx())
        
        for j, predicted_shift in enumerate(logits):
            target_shift = labels[j]
            output["molecule"].append(smiles)
            output["target_shift"].append(tf.get_static_value(target_shift)[0])
            output["predicted_shift"].append(tf.get_static_value(predicted_shift)[0])
            mae = tf.math.reduce_mean(tf.keras.losses.MeanAbsoluteError().call(y_true=target_shift, y_pred=predicted_shift))
            output["mae"].append(tf.get_static_value(mae)) 

    
    output_df = pd.DataFrame.from_dict(output)
    print(output_df)
    output_df.to_csv(path + "data/own_data/shift_results.csv.gz", compression='gzip')


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
    
    output["molecule"].append(tf.get_static_value(molecule).astype(str))
    output["mae"].append(tf.get_static_value(mae))
    output["reverse_mae"].append(tf.get_static_value(reverse_mae))

    output_df = pd.DataFrame.from_dict(output)
    print(output["molecule"])
    print("Molecule labels")
    print(labels)
    print("Model shifts")
    print(logits)
    print(output_df)

def plot_results(path):
    results = pd.read_csv(path + "data/own_data/data_results.csv.gz", index_col=0)
    fig = px.scatter(results, x=results.index, y='mae')
    '''fig.add_trace(go.Scatter(x=results.index, y=results['mae'], mode='markers', name='mae'))
    fig.add_trace(go.Scatter(x=results.index, y=results['reverse_mae'], mode='markers', name='reverse_mae'))'''
    fig.update_layout(xaxis=dict(type='category'))
    plot = DynamicPlot(fig)
    plot.show()

def plot_shift_errors(path):
    results = pd.read_csv(path + "data/own_data/shift_results.csv.gz", index_col=0)
    fig = px.histogram(results, x='target_shift', y='mae', nbins=50, histfunc='avg')
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
    results = pd.read_csv(path + "data/own_data/shift_results.csv.gz", index_col=0)
    results["reverse_better"] =  np.where(results['reverse_mae'] < results["mae"], "yes", "no")
    print("Average MAE:\n" + str(results.loc[:, "mae"].mean()))
    print("Number of samples where reverse is better: " + str(results["reverse_better"].value_counts()[1]))
    print("Samples where reverse is better: " + str(results[results["reverse_better"]=="yes"]))
    print("Number of samples where MAE > 1: " + str(len(results[results["mae"]>1])))
    print("Samples where MAE > 1:\n" + str(results[results["mae"]>1]))

def visualize_errors(path):
    results = pd.read_csv(path + "data/own_data/shift_results.csv.gz", index_col=0)
    print(results)
    smiles = results["molecule"][0]
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    d = Draw.rdMolDraw2D.MolDraw2DCairo(500, 500)
    atom_data = results.loc[results["molecule"] == smiles]
    highlightAtoms = []
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        matching_data = atom_data.loc[atom_data["index"] == index]
        if not matching_data.empty:
            atom.SetProp('atomNote', str(matching_data["mae"].values[0]))
            highlightAtoms.append(index)

    colours=[(1.0, 0.0, 0.0),(0.0, 1.0, 0.0)]
    atom_cols = {}
    for i, atom in enumerate(highlightAtoms):
        atom_cols[atom] = colours[i]

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=highlightAtoms, highlightAtomColors=atom_cols)
    d.FinishDrawing()
    d.WriteDrawingText(path + "mol.png")


if __name__ == "__main__":
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    #path = "/home1/s3665828/code/CASCADE/"
    path = "C:/Users/niels/Documents/repo/CASCADE/"

    graph_schema = tfgnn.read_schema(path + "code/predicting_model/GraphSchema.pbtxt")
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    model = tf.saved_model.load(path + "code/predicting_model/Shift/DFTNN/gnn/models/DFT_model")
    signature_fn = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    
    dataset_provider = runner.TFRecordDatasetProvider(filenames=[path + "data/own_data/shift/DFT_train.tfrecords"])
    dataset = dataset_provider.get_dataset(tf.distribute.InputContext())

    visualize_errors(path)

    #evaluate_model_shifts(dataset, model, path)
    #evaluate_model(dataset, model)
    #evaluate_sample(dataset, model, 7)
    #plot_shift_errors(path)