import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn
from rdkit import Chem
from rdkit.Chem import AllChem

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


def processData(filepath):
    file = open(filepath, 'r')
    text = file.read()
    samples = text.split('\n\n')
    mol_id = 0

    mol_dict = {"mol_id":[], "smiles":[], "n_atoms":[], "n_bonds":[], "n_pro":[]}
    atom_dict = {"mol_id":[], "atom_symbol":[], "Shift":[]}
    bond_dict = {"mol_id":[], "bond_type":[], "distance":[], "source":[], "target":[]}

    for sample in tqdm(samples):
        sampleSplit = sample.split("\n")
        if sampleSplit == ['']:
            continue

        smiles = sampleSplit[0]

        if (smiles == ""):
            continue

        smilesH = sampleSplit[1]
        mol = Chem.MolFromSmiles(smiles)
        n_atoms = mol.GetNumAtoms(onlyExplicit=False)
        n_bonds = mol.GetNumBonds(onlyHeavy=False)

        mol_dict["mol_id"].append(mol_id)
        mol_dict["smiles"].append(smiles)
        mol_dict["n_atoms"].append(n_atoms)
        mol_dict["n_bonds"].append(n_bonds)
        mol_dict["n_pro"].append(n_atoms - mol.GetNumHeavyAtoms())

        iter_H = 0
        mol = Chem.AddHs(mol)
        mol, embedding_found = findEmbedding(mol)
        if (embedding_found):
            try:
                distance_matrix = Chem.Get3DDistanceMatrix(mol)
            except ValueError:
                bad_mol = Chem.RemoveHs(mol)
                print(Chem.MolToSmiles(bad_mol))
    
            for n, atom in enumerate(mol.GetAtoms()):
                atom_dict["mol_id"].append(mol_id)
                atom_symbol = atom.GetSymbol()
                atom_dict["atom_symbol"].append(atom_symbol)
                
                if (atom_symbol == "H"):
                    atomSplit = sampleSplit[3+iter_H].split(",")
                    atom_dict["Shift"].append(atomSplit[1])

                    iter_H += 1
                else:
                    atom_dict["Shift"].append(0.0)  #0.0 shift for non-H atoms

            for n, bond in enumerate(mol.GetBonds()):
                bond_dict["mol_id"].append(mol_id)
                bond_dict["bond_type"].append(bond.GetBondTypeAsDouble())

                bond_source = bond.GetBeginAtomIdx()
                bond_target = bond.GetEndAtomIdx()
                bond_dict["distance"].append(distance_matrix[bond_source][bond_target])
                bond_dict["source"].append(bond_source)
                bond_dict["target"].append(bond_target)

            mol_id += 1

    mol_df = pd.DataFrame.from_dict(mol_dict)
    atom_df = pd.DataFrame.from_dict(atom_dict)
    bond_df = pd.DataFrame.from_dict(bond_dict)

    print(mol_df)
    print(atom_df)
    print(bond_df)

    mol_df.to_csv("code/predicting_model/Shift/DFTNN/own_data_mol.csv.gz", compression='gzip')
    atom_df.to_csv("code/predicting_model/Shift/DFTNN/own_data_atom.csv.gz", compression='gzip')
    bond_df.to_csv("code/predicting_model/Shift/DFTNN/own_data_bond.csv.gz", compression='gzip')


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
        indices = np.append(indices, index)

    return tf.one_hot(tf.convert_to_tensor(indices), 4)

def create_graph_tensor(mol_data, atom_data, bond_data):
    H_indices = atom_data.index[atom_data["atom_symbol"] == "H"].tolist()
    atom_nums = one_hot_encode_atoms(atom_data["atom_symbol"])

    graph_tensor = tfgnn.GraphTensor.from_pieces(

        context = tfgnn.Context.from_fields(features = {"smiles": mol_data["smiles"]}),

        node_sets = {
            "atom": tfgnn.NodeSet.from_fields(
                sizes = mol_data["n_atoms"],
                features = {"atom_num": atom_nums}
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
                features = {"bond_type": bond_data["bond_type"],
                            "distance": bond_data["distance"]}
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

if __name__ == "__main__":
    # create dataframes if they do not exist yet
    if not os.path.isfile("code/predicting_model/Shift/DFTNN/own_data_mol.csv.gz"):
        processData('data/own_data/cleaned_full_dataset.txt')

    mol_df = pd.read_csv("code/predicting_model/Shift/DFTNN/own_data_mol.csv.gz", index_col=0)
    atom_df = pd.read_csv("code/predicting_model/Shift/DFTNN/own_data_atom.csv.gz", index_col=0)
    bond_df = pd.read_csv("code/predicting_model/Shift/DFTNN/own_data_bond.csv.gz", index_col=0)

    mol_df = mol_df.sample(frac=1)  #shuffle

    train_data = tf.io.TFRecordWriter("data/own_data/shift_train.tfrecords")
    test_data = tf.io.TFRecordWriter("data/own_data/shift_test.tfrecords")
    valid_data = tf.io.TFRecordWriter("data/own_data/shift_valid.tfrecords")

    total = len(mol_df)
    '''print(len(mol_df))
    print(len(mol_df)*0.70)
    print(len(mol_df)*0.15)'''

    for idx, mol_id in tqdm(enumerate(mol_df["mol_id"])):
        mol_data = mol_df.loc[mol_df["mol_id"] == mol_id]
        atom_data = atom_df.loc[atom_df["mol_id"] == mol_id]
        bond_data = bond_df.loc[bond_df["mol_id"] == mol_id]

        if (mol_data['n_pro'].values == 0):
            continue
        
        # make sure the index resets to 0 instead of continuing from the previous molecule
        atom_data = atom_data.reset_index(drop=True)
            
        graph_tensor = create_graph_tensor(mol_data, atom_data, bond_data)
        example = tfgnn.write_example(graph_tensor)

        if idx < total*0.7:
            train_data.write(example.SerializeToString())
        elif idx < total*0.85:
            valid_data.write(example.SerializeToString())
        else:
            test_data.write(example.SerializeToString())