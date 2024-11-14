import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn
from rdkit import Chem
from rdkit.Chem import AllChem

import gzip
import os

from tqdm import tqdm

def findEmbedding(mol):
    attempts = 0
    flag = 0

    while(flag == 0 and attempts <= 500):   # Attempt to find a proper embedding 500 times
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(mol)
            flag = 1
        except:
            attempts += 1

        if (flag == 0):
            bad_mol = Chem.RemoveHs(mol)
            print("Error with optimizing")
            print(Chem.MolToSmiles(bad_mol))
            print(attempts)
            continue

        try:
            Chem.rdMolTransforms.CanonicalizeMol(mol, normalizeCovar=True, ignoreHs=False)
        except ValueError:
            bad_mol = Chem.RemoveHs(mol)
            print("Error with canonicalizing")
            print(Chem.MolToSmiles(bad_mol))
            continue

    return mol, flag


def create_dictionary(key, path, type, save=False, filepath="", name="", smiles=""):
    mol_list = []
    atom_list = []
    bond_list = []
    distance_list = []

    if key == 0:    # own data
        file = open(path + filepath, 'r')
        text = file.read()
        samples = text.split('\n\n')
        for sample in tqdm(samples):
            sampleSplit = sample.split("\n")
            if sampleSplit == ['']:
                continue

            mol_id = sampleSplit[0]
            smiles = sampleSplit[1]
            mol = Chem.MolFromSmiles(smiles)
            mol_entry, atom_entry, bond_entry, distance_entry = fill_dictionary(key, mol_id, mol, shift_data=sampleSplit)
            mol_list.extend(mol_entry)
            atom_list.extend(atom_entry)
            bond_list.extend(bond_entry)
            distance_list.extend(distance_entry)

    elif key == 1:    # DFT data
        with gzip.open(path + "data/DFT8K/DFT.sdf.gz", 'rb') as dft:
            shift_df = pd.read_csv(path + "data/DFT8K/DFT8K.csv.gz", index_col=0)
            mol_suppl = Chem.ForwardSDMolSupplier(dft, sanitize=False, removeHs=False)
            #mol_suppl = Chem.ForwardSDMolSupplier(dft, sanitize=True, removeHs=False)
            for mol in mol_suppl:
                mol_id = int(mol.GetProp("_Name"))
                mol.UpdatePropertyCache()
                mol_entry, atom_entry, bond_entry, distance_entry = fill_dictionary(key, mol_id, mol, shift_data=shift_df)
                mol_list.extend(mol_entry)
                atom_list.extend(atom_entry)
                bond_list.extend(bond_entry)
                distance_list.extend(distance_entry)

    elif key == 2:  # single molecule
        mol = Chem.MolFromSmiles(smiles)
        mol_entry, atom_entry, bond_entry, distance_entry = fill_dictionary(key, 0, mol)
        mol_list.extend(mol_entry)
        atom_list.extend(atom_entry)
        bond_list.extend(bond_entry)
        distance_list.extend(distance_entry)

    # convert the list of dicts to a single dict
    mol_dict = {}
    for k in mol_list[0].keys():
        mol_dict[k] = tuple(mol_dict[k] for mol_dict in mol_list)

    atom_dict = {}
    for k in atom_entry[0].keys():
        atom_dict[k] = tuple(atom_dict[k] for atom_dict in atom_list)

    bond_dict = {}
    for k in bond_list[0].keys():
        bond_dict[k] = tuple(bond_dict[k] for bond_dict in bond_list)

    distance_dict = {}
    for k in distance_list[0].keys():
        distance_dict[k] = tuple(distance_dict[k] for distance_dict in distance_list)

    # convert the dicts to pandas dataframes
    mol_df = pd.DataFrame.from_dict(mol_dict)
    atom_df = pd.DataFrame.from_dict(atom_dict)
    bond_df = pd.DataFrame.from_dict(bond_dict)
    distance_df = pd.DataFrame.from_dict(distance_dict)

    print(mol_df)
    print(atom_df)
    print(bond_df)
    print(distance_df)

    bond_df["norm_distance"] = (bond_df["distance"] - bond_df["distance"].mean())/bond_df["distance"].std()

    if save:
        mol_df.to_csv(path + "code/predicting_model/" + type + "/" + name + "_mol.csv.gz", compression='gzip')
        atom_df.to_csv(path + "code/predicting_model/" + type + "/" + name + "_atom.csv.gz", compression='gzip')
        bond_df.to_csv(path + "code/predicting_model/" + type + "/" + name + "_bond.csv.gz", compression='gzip')
        distance_df.to_csv(path + "code/predicting_model/" + type + "/" + name + "_distance.csv.gz", compression='gzip')

    return mol_df, atom_df, bond_df, distance_df
    

def fill_dictionary(key, mol_id, mol, shift_data=""):
    cutoff_distance = 5

    mol_list = []
    atom_list = []
    bond_list = []
    distance_list = []

    mol_dict = {"mol_id":[], "smiles":[], "n_atoms":[], "n_bonds":[], "n_pro":[]}
    
    n_atoms = mol.GetNumAtoms(onlyExplicit=False)
    n_bonds = mol.GetNumBonds(onlyHeavy=False)

    mol_dict["mol_id"] = mol_id
    mol_dict["smiles"] = Chem.MolToSmiles(mol, canonical = False)
    mol_dict["n_atoms"] = n_atoms
    mol_dict["n_bonds"] = n_bonds
    mol_dict["n_pro"] = n_atoms - mol.GetNumHeavyAtoms()

    mol_list.append(mol_dict)

    embedding_found = 0
    try:    # check for an existing embedding
        mol.GetConformer()
        embedding_found = 1
    except ValueError:  # make a new embedding
        mol, embedding_found = findEmbedding(mol)
        mol = Chem.AddHs(mol, addCoords=True)
    
    if (embedding_found):
        try:
            distance_matrix = Chem.Get3DDistanceMatrix(mol)
        except ValueError:
            bad_mol = Chem.RemoveHs(mol)
            print(Chem.MolToSmiles(bad_mol))

        iter_H = 0

        for n, atom in enumerate(mol.GetAtoms()):
            atom_dict = {"mol_id":[], "atom_idx":[], "atom_symbol":[], "chiral_tag":[], "degree":[], 
                 "formal_charge":[], "hybridization":[], "is_aromatic":[], 
                 "no_implicit":[], "num_Hs":[], "valence":[], "Shift":[], "Shape":[]}

            atom_dict["mol_id"] = mol_id
            atom_dict["atom_idx"] = atom.GetIdx()
            atom_symbol = atom.GetSymbol()
            atom_dict["atom_symbol"] = atom_symbol
            atom_dict["chiral_tag"] = atom.GetChiralTag()
            atom_dict["degree"] = atom.GetTotalDegree()
            atom_dict["formal_charge"] = atom.GetFormalCharge() + 1   # Now ranges from 0...2 instead of -1...1
            atom_dict["hybridization"] = atom.GetHybridization()
            atom_dict["is_aromatic"] = atom.GetIsAromatic()
            atom_dict["no_implicit"] = atom.GetNoImplicit()
            atom_dict["valence"] = atom.GetTotalValence()
            atom_dict["num_Hs"] = atom.GetTotalNumHs(includeNeighbors=True)
            
            if (atom_symbol == "H"):
                if key == 0:    # using our own data, with shape data
                    atomSplit = shift_data[4+iter_H].split(",")
                    atom_dict["Shift"] = atomSplit[1]
                    atom_dict["Shape"] = atomSplit[3]
                    iter_H += 1
                elif key == 1:  # using DFT data
                    atom_dict["Shift"] = shift_data.loc[(shift_data['mol_id'] == mol_id) & (shift_data['atom_index'] == n)]["Shift"].values[0]
                    atom_dict["Shape"] = "-"
                elif key == 2: # using molecules without labels (for making new predictions)
                    atom_dict["Shift"] = 0.0
                    atom_dict["Shape"] = "-"
                
            else: # not really necessary now, but useful for adding C-NMR in the future
                atom_dict["Shift"] = 0.0  # 0.0 shift for non-H atoms
                atom_dict["Shape"] = "-"

            atom_list.append(atom_dict)

            for target, distance in enumerate(distance_matrix[n]):
                distance_dict = {"mol_id":[], "distance":[], "source":[], "target":[]}
                distance_dict["mol_id"] = mol_id
                distance_dict["source"] = n
                distance_dict["target"] = target
                distance_dict["distance"] = distance
                if distance != 0 and distance < cutoff_distance:
                    distance_list.append(distance_dict)
            

        for n, bond in enumerate(mol.GetBonds()):
            bond_dict = {"mol_id":[], "bond_type":[], "distance":[], "is_conjugated":[], 
                 "stereo":[], "source":[], "target":[]}
            
            bond_dict["mol_id"] = mol_id
            bond_dict["bond_type"] = bond.GetBondType()

            bond_source = bond.GetBeginAtomIdx()
            bond_target = bond.GetEndAtomIdx()
            bond_dict["distance"] = distance_matrix[bond_source][bond_target]
            bond_dict["source"] = bond_source
            bond_dict["target"] = bond_target
            bond_dict["is_conjugated"] = bond.GetIsConjugated()
            bond_dict["stereo"] = bond.GetStereo()

            bond_list.append(bond_dict)

    return mol_list, atom_list, bond_list, distance_list
    

def one_hot_encode_atoms(atom_symbols):
    indices = np.empty(0, dtype=int)
    for symbol in atom_symbols:
        if symbol == "H":
            index = 0
        elif symbol == "C":
            index = 1
        elif symbol == "O":
            index = 2
        elif symbol == "N":
            index = 3
        elif symbol == "F":
            index = 4
        elif symbol == "Cl":
            index = 5
        elif symbol == "S":
            index = 6
        elif symbol == "P":
            index = 7        
        indices = np.append(indices, index)

    return tf.one_hot(tf.convert_to_tensor(indices), 7)


def one_hot_encode_shape(shape_symbols):            # The output will be a matrix of four one-hots, where each one-hot encodes for a shape
    one_hot_shapes = np.empty((0,4,8), dtype=int)   # i.e. 'dtp' will be encoded as matrix of four one-hots where the first one encodes
    for shape in shape_symbols:                     # for 'd', the second for 't', the third for 'p', and the rest for 's'
        indices = np.ones(4, dtype=int)
        for i in range(len(shape)):
            if shape[i] == '-': 
                indices *= -1  # only for non-H atoms
                break   
            elif shape[i] == 'm': indices[i] = 0
            elif shape[i] == 's': indices[i] = 1
            elif shape[i] == 'd': indices[i] = 2
            elif shape[i] == 't': indices[i] = 3
            elif shape[i] == 'q': indices[i] = 4
            elif shape[i] == 'p': indices[i] = 5
            elif shape[i] == 'h': indices[i] = 6
            elif shape[i] == 'v': indices[i] = 7
            
        one_hot_shapes = np.append(one_hot_shapes, np.expand_dims(tf.one_hot(tf.convert_to_tensor(indices), 8), axis=0), axis=0)
    return one_hot_shapes


def create_graph_tensor_shift(mol_data, atom_data, bond_data, distance_data):
    H_indices = atom_data.index[atom_data["atom_symbol"] == "H"].tolist()
    atom_syms = one_hot_encode_atoms(atom_data["atom_symbol"])
    chiral_tags = tf.one_hot(atom_data["chiral_tag"], 9)
    hybridizations = tf.one_hot(atom_data["hybridization"], 9)
    stereo = tf.one_hot(bond_data["stereo"], 8)
    bond_type = tf.one_hot(bond_data["bond_type"], 22)

    graph_tensor = tfgnn.GraphTensor.from_pieces(

        context = tfgnn.Context.from_fields(features = {"smiles": mol_data["smiles"],
                                                        "_mol_id": mol_data["mol_id"]}),

        node_sets = {
            "atom": tfgnn.NodeSet.from_fields(
                sizes = mol_data["n_atoms"],
                features = {"_atom_idx": atom_data["atom_idx"],
                            "atom_sym": atom_syms,
                            "chiral_tag": chiral_tags,
                            "degree": atom_data["degree"],
                            "formal_charge": atom_data["formal_charge"],
                            "hybridization": hybridizations,
                            "is_aromatic": atom_data["is_aromatic"].astype(int),
                            "no_implicit": atom_data["no_implicit"].astype(int),
                            "num_Hs": atom_data["num_Hs"],
                            "valence": atom_data["valence"]}
            ),
            "_readout": tfgnn.NodeSet.from_fields(
                sizes = mol_data["n_pro"],
                features = {"shift": atom_data["Shift"][H_indices]}
            )
        },

        edge_sets = {
            "bond": tfgnn.EdgeSet.from_fields(
                sizes = mol_data["n_bonds"],
                adjacency = tfgnn.Adjacency.from_indices(
                    source = ("atom", bond_data["source"]),
                    target = ("atom", bond_data["target"])),
                features = {"bond_type": bond_type,
                            "distance": bond_data["distance"].astype('float32'),
                            "is_conjugated": bond_data["is_conjugated"].astype(int),
                            "stereo": stereo,
                            "normalized_distance": bond_data["norm_distance"]}
            ),
            "interatomic_distance": tfgnn.EdgeSet.from_fields(
                sizes = mol_data["n_distance"],
                adjacency = tfgnn.Adjacency.from_indices(
                    source = ("atom", distance_data["source"]),
                    target = ("atom", distance_data["target"])),
                features = {"distance": distance_data["distance"].astype('float32')}
            ),
            "_readout/shift": tfgnn.EdgeSet.from_fields(
                sizes = mol_data["n_pro"],
                adjacency = tfgnn.Adjacency.from_indices(
                    source = ("atom", H_indices),
                    target = ("_readout", list(range(mol_data["n_pro"].values[0])))
                )
            )
        }
    )

    return graph_tensor


def create_graph_tensor_shape(mol_data, atom_data, bond_data, distance_data):
    H_indices = atom_data.index[atom_data["atom_symbol"] == "H"].tolist()
    atom_syms = one_hot_encode_atoms(atom_data["atom_symbol"])
    chiral_tags = tf.one_hot(atom_data["chiral_tag"], 9)
    hybridizations = tf.one_hot(atom_data["hybridization"], 9)
    stereo = tf.one_hot(bond_data["stereo"], 8)
    bond_type = tf.one_hot(bond_data["bond_type"], 22)
    shape = one_hot_encode_shape(atom_data["Shape"])

    graph_tensor = tfgnn.GraphTensor.from_pieces(

        context = tfgnn.Context.from_fields(features = {"smiles": mol_data["smiles"],
                                                        "_mol_id": mol_data["mol_id"]}),

        node_sets = {
            "atom": tfgnn.NodeSet.from_fields(
                sizes = mol_data["n_atoms"],
                features = {"atom_sym": atom_syms,
                            "chiral_tag": chiral_tags,
                            "degree": atom_data["degree"],
                            "formal_charge": atom_data["formal_charge"],
                            "hybridization": hybridizations,
                            "is_aromatic": atom_data["is_aromatic"].astype(int),
                            "no_implicit": atom_data["no_implicit"].astype(int),
                            "num_Hs": atom_data["num_Hs"],
                            "valence": atom_data["valence"],
                            "shift": atom_data["Shift"]}
            ),
            "_readout": tfgnn.NodeSet.from_fields(
                sizes = mol_data["n_pro"],
                features = {"shape": shape[H_indices]}
            )
        },

        edge_sets = {
            "bond": tfgnn.EdgeSet.from_fields(
                sizes = mol_data["n_bonds"],
                adjacency = tfgnn.Adjacency.from_indices(
                    source = ("atom", bond_data["source"]),
                    target = ("atom", bond_data["target"])),
                features = {"bond_type": bond_type,
                            "distance": bond_data["distance"].astype('float32'),
                            "is_conjugated": bond_data["is_conjugated"].astype(int),
                            "stereo": stereo,
                            "normalized_distance": bond_data["norm_distance"]}
            ),
            "interatomic_distance": tfgnn.EdgeSet.from_fields(
                sizes = mol_data["n_distance"],
                adjacency = tfgnn.Adjacency.from_indices(
                    source = ("atom", distance_data["source"]),
                    target = ("atom", distance_data["target"])),
                features = {"distance": distance_data["distance"].astype('float32')}
            ),
            "_readout/shape": tfgnn.EdgeSet.from_fields(
                sizes = mol_data["n_pro"],
                adjacency = tfgnn.Adjacency.from_indices(
                    source = ("atom", H_indices),
                    target = ("_readout", list(range(mol_data["n_pro"].values[0])))
                )
            )
        }
    )

    return graph_tensor

def create_tensors(path, name, type="Shift"):
    mol_df = pd.read_csv(path + "code/predicting_model/" + type + "/" + name + "_mol.csv.gz", index_col=0)
    atom_df = pd.read_csv(path + "code/predicting_model/" + type + "/" + name + "_atom.csv.gz", index_col=0)
    bond_df = pd.read_csv(path + "code/predicting_model/" + type + "/" + name + "_bond.csv.gz", index_col=0)
    distance_df = pd.read_csv(path + "code/predicting_model/" + type + "/" + name + "_distance.csv.gz", index_col=0)

    train_data = tf.io.TFRecordWriter(path + "data/own_data/" + type + "/" + name + "_train.tfrecords")
    test_data = tf.io.TFRecordWriter(path + "data/own_data/" + type + "/" + name + "_test.tfrecords")
    valid_data = tf.io.TFRecordWriter(path + "data/own_data/" + type + "/" + name + "_valid.tfrecords")
    all_data = tf.io.TFRecordWriter(path + "data/own_data/" + type + "/all_" + name + "_data.tfrecords")

    total = len(mol_df)
    n_train = math.floor(total*0.7)
    n_valid = math.floor(total*0.15)
    n_test = total - n_train - n_valid
    print("Total dataset size: " + str(total))
    print("Training set size: " + str(n_train))
    print("Validation set size: " + str(n_valid))
    print("Testing set size: " + str(n_test))

    for idx, mol_id in tqdm(enumerate(mol_df["mol_id"])):
        mol_data = mol_df.loc[mol_df["mol_id"] == mol_id]
        atom_data = atom_df.loc[atom_df["mol_id"] == mol_id]
        bond_data = bond_df.loc[bond_df["mol_id"] == mol_id]
        distance_data = distance_df.loc[distance_df["mol_id"] == mol_id]

        if (mol_data['n_pro'].values == 0):
            continue
        
        mol_data.insert(5, "n_distance", len(distance_data["distance"]))
        # make sure the index resets to 0 instead of continuing from the previous molecule
        atom_data = atom_data.reset_index(drop=True)
        if type == "Shift":
            graph_tensor = create_graph_tensor_shift(mol_data, atom_data, bond_data, distance_data)
        elif type == "Shape":
            graph_tensor = create_graph_tensor_shape(mol_data, atom_data, bond_data, distance_data)
        elif type == "Couplings":
            print("not implemented yet")
            #graph_tensor = create_graph_tensor_couplings(mol_data, atom_data, bond_data)
        example = tfgnn.write_example(graph_tensor)
        
        all_data.write(example.SerializeToString())
        if idx < n_train:
            train_data.write(example.SerializeToString())
        elif idx < (n_train+n_valid):
            valid_data.write(example.SerializeToString())
        else:
            test_data.write(example.SerializeToString())

def create_single_tensor(mol_data, atom_data, bond_data, type="Shift"):
    if type == "Shift":
        graph_tensor = create_graph_tensor_shift(mol_data, atom_data, bond_data)
    elif type == "Shape":
        graph_tensor = create_graph_tensor_shape(mol_data, atom_data, bond_data)
    elif type == "Couplings":
        print("not implemented yet")

    example = tfgnn.write_example(graph_tensor)
    return example.SerializeToString()

def process_samples(key, path, type="Shift", save=False, file="", name="", smiles=""):
    # key 0 = process own data (txt)
    # key 1 = process DFT data (sdf + csv)
    # key 2 = process single sample (smiles)
    if (key == 0):
        create_dictionary(key, path, type, save, file, name)
        create_tensors(path, name, type)
    elif(key == 1):
        create_dictionary(key, path, type="Shift", save=save, name="DFT") 
        create_tensors(path, name="DFT", type="Shift") # DFT data can only create shift tensors
    elif(key == 2):
        mol_df, atom_df, bond_df = create_dictionary(key, path, smiles=smiles, name=smiles)
        return create_single_tensor(mol_df, atom_df, bond_df)


if __name__ == "__main__":
    path = "/home1/s3665828/code/CASCADE/"
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    #path = "C:/Users/niels/Documents/repo/CASCADE/"

    process_samples(1, path, file="data/own_data/own_data_with_id.txt", save=True, name="own", type="Shape")