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


def processData(filepath, path):
    file = open(filepath, 'r')
    text = file.read()
    samples = text.split('\n\n')
    mol_id = 0

    mol_dict = {"mol_id":[], "smiles":[], "n_atoms":[], "n_bonds":[], "n_pro":[]}
    atom_dict = {"mol_id":[], "atom_symbol":[], "chiral_tag":[], "degree":[], 
                 "formal_charge":[], "hybridization":[], "is_aromatic":[], 
                 "no_implicit":[], "num_Hs":[], "valence":[], "Shift":[], "Shape":[]}
    bond_dict = {"mol_id":[], "bond_type":[], "distance":[], "is_conjugated":[], 
                 "stereo":[], "source":[], "target":[]}

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
        mol, embedding_found = findEmbedding(mol)
        mol = Chem.AddHs(mol, addCoords=True)
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
                atom_dict["chiral_tag"].append(atom.GetChiralTag())
                atom_dict["degree"].append(atom.GetTotalDegree())
                atom_dict["formal_charge"].append(atom.GetFormalCharge() + 1)   # Now ranges from 0...2 instead of -1...1
                atom_dict["hybridization"].append(atom.GetHybridization())
                atom_dict["is_aromatic"].append(atom.GetIsAromatic())
                atom_dict["no_implicit"].append(atom.GetNoImplicit())
                atom_dict["valence"].append(atom.GetTotalValence())
                atom_dict["num_Hs"].append(atom.GetTotalNumHs(includeNeighbors=True))
                
                if (atom_symbol == "H"):
                    atomSplit = sampleSplit[3+iter_H].split(",")
                    atom_dict["Shift"].append(atomSplit[1])
                    atom_dict["Shape"].append(atomSplit[3])
                    iter_H += 1
                else: # not really necessary now, but useful for adding C-NMR in the future
                    atom_dict["Shift"].append(0.0)  # 0.0 shift for non-H atoms
                    atom_dict["Shape"].append("-")  

            for n, bond in enumerate(mol.GetBonds()):
                bond_dict["mol_id"].append(mol_id)
                bond_dict["bond_type"].append(bond.GetBondType())

                bond_source = bond.GetBeginAtomIdx()
                bond_target = bond.GetEndAtomIdx()
                bond_dict["distance"].append(distance_matrix[bond_source][bond_target])
                bond_dict["source"].append(bond_source)
                bond_dict["target"].append(bond_target)
                bond_dict["is_conjugated"].append(bond.GetIsConjugated())
                bond_dict["stereo"].append(bond.GetStereo())

            mol_id += 1


    mol_df = pd.DataFrame.from_dict(mol_dict)
    atom_df = pd.DataFrame.from_dict(atom_dict)
    bond_df = pd.DataFrame.from_dict(bond_dict)

    bond_df["norm_distance"] = (bond_df["distance"] - bond_df["distance"].mean())/bond_df["distance"].std()

    mol_df.to_csv(path + "code/predicting_model/Shift/DFTNN/small_data_mol.csv.gz", compression='gzip')
    atom_df.to_csv(path + "code/predicting_model/Shift/DFTNN/small_data_atom.csv.gz", compression='gzip')
    bond_df.to_csv(path + "code/predicting_model/Shift/DFTNN/small_data_bond.csv.gz", compression='gzip')

def processDFTData(path):

    #print(shift_df.loc[(shift_df['mol_id'] == 2192)]["Shift"])

    mol_dict = {"mol_id":[], "smiles":[], "n_atoms":[], "n_bonds":[], "n_pro":[]}
    atom_dict = {"mol_id":[], "atom_symbol":[], "chiral_tag":[], "degree":[], 
                 "formal_charge":[], "hybridization":[], "is_aromatic":[], 
                 "no_implicit":[], "num_Hs":[], "valence":[], "Shift":[]}
    bond_dict = {"mol_id":[], "bond_type":[], "distance":[], "is_conjugated":[], 
                 "stereo":[], "source":[], "target":[]}

    with gzip.open(path + "data/DFT8K/DFT.sdf.gz", 'rb') as dft:
        shift_df = pd.read_csv(path + "data/DFT8K/DFT8K.csv.gz", index_col=0)
        mol_suppl = Chem.ForwardSDMolSupplier(dft, removeHs=False)
        for mol in mol_suppl:
            n_atoms = mol.GetNumAtoms(onlyExplicit=False)
            n_bonds = mol.GetNumBonds(onlyHeavy=False)
            mol_id = int(mol.GetProp("_Name"))

            mol_dict["mol_id"].append(mol_id)
            mol_dict["smiles"].append(Chem.MolToSmiles(mol))
            mol_dict["n_atoms"].append(n_atoms)
            mol_dict["n_bonds"].append(n_bonds)
            mol_dict["n_pro"].append(n_atoms - mol.GetNumHeavyAtoms())

            iter_H = 0
            distance_matrix = Chem.Get3DDistanceMatrix(mol)
        
            for n, atom in enumerate(mol.GetAtoms()):
                atom_dict["mol_id"].append(mol_id)
                atom_symbol = atom.GetSymbol()
                atom_dict["atom_symbol"].append(atom_symbol)
                atom_dict["chiral_tag"].append(atom.GetChiralTag())
                atom_dict["degree"].append(atom.GetTotalDegree())
                atom_dict["formal_charge"].append(atom.GetFormalCharge() + 1)   # Now ranges from 0...2 instead of -1...1
                atom_dict["hybridization"].append(atom.GetHybridization())
                atom_dict["is_aromatic"].append(atom.GetIsAromatic())
                atom_dict["no_implicit"].append(atom.GetNoImplicit())
                atom_dict["valence"].append(atom.GetTotalValence())
                atom_dict["num_Hs"].append(atom.GetTotalNumHs(includeNeighbors=True))
                
                if (atom_symbol == "H"):
                    atom_dict["Shift"].append(shift_df.loc[(shift_df['mol_id'] == mol_id) & (shift_df['atom_index'] == n)]["Shift"].values[0])
                else: # not really necessary now, but useful for adding C-NMR in the future
                    atom_dict["Shift"].append(0.0)  # 0.0 shift for non-H atoms

            for n, bond in enumerate(mol.GetBonds()):
                bond_dict["mol_id"].append(mol_id)
                bond_dict["bond_type"].append(bond.GetBondType())

                bond_source = bond.GetBeginAtomIdx()
                bond_target = bond.GetEndAtomIdx()
                bond_dict["distance"].append(distance_matrix[bond_source][bond_target])
                bond_dict["source"].append(bond_source)
                bond_dict["target"].append(bond_target)
                bond_dict["is_conjugated"].append(bond.GetIsConjugated())
                bond_dict["stereo"].append(bond.GetStereo())


    mol_df = pd.DataFrame.from_dict(mol_dict)
    atom_df = pd.DataFrame.from_dict(atom_dict)
    bond_df = pd.DataFrame.from_dict(bond_dict)

    bond_df["norm_distance"] = (bond_df["distance"] - bond_df["distance"].mean())/bond_df["distance"].std()

    mol_df.to_csv(path + "code/predicting_model/Shift/DFTNN/DFT_data_mol.csv.gz", compression='gzip')
    atom_df.to_csv(path + "code/predicting_model/Shift/DFTNN/DFT_data_atom.csv.gz", compression='gzip')
    bond_df.to_csv(path + "code/predicting_model/Shift/DFTNN/DFT_data_bond.csv.gz", compression='gzip')


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

def one_hot_encode_shape(shape_symbols):            # The output will be a matrix of six one-hots, where each one-hot encodes for a shape
    one_hot_shapes = np.empty((0,6,8), dtype=int)   # i.e. 'dtp' will be encoded as matrix of six one-hots where the first one encodes
    for shape in shape_symbols:                     # for 'd', the second for 't', the third for 'p', and the rest for 's'
        indices = np.ones(6, dtype=int)
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

def create_graph_tensor_shift(mol_data, atom_data, bond_data):
    H_indices = atom_data.index[atom_data["atom_symbol"] == "H"].tolist()
    atom_nums = one_hot_encode_atoms(atom_data["atom_symbol"])
    chiral_tags = tf.one_hot(atom_data["chiral_tag"], 9)
    hybridizations = tf.one_hot(atom_data["hybridization"], 9)
    stereo = tf.one_hot(bond_data["stereo"], 8)
    bond_type = tf.one_hot(bond_data["bond_type"], 22)

    graph_tensor = tfgnn.GraphTensor.from_pieces(

        context = tfgnn.Context.from_fields(features = {"smiles": mol_data["smiles"]}),

        node_sets = {
            "atom": tfgnn.NodeSet.from_fields(
                sizes = mol_data["n_atoms"],
                features = {"atom_num": atom_nums,
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

def create_graph_tensor_shape(mol_data, atom_data, bond_data):
    H_indices = atom_data.index[atom_data["atom_symbol"] == "H"].tolist()
    atom_nums = one_hot_encode_atoms(atom_data["atom_symbol"])
    chiral_tags = tf.one_hot(atom_data["chiral_tag"], 9)
    hybridizations = tf.one_hot(atom_data["hybridization"], 9)
    stereo = tf.one_hot(bond_data["stereo"], 8)
    bond_type = tf.one_hot(bond_data["bond_type"], 22)
    shape = one_hot_encode_shape(atom_data["Shape"])

    graph_tensor = tfgnn.GraphTensor.from_pieces(

        context = tfgnn.Context.from_fields(features = {"smiles": mol_data["smiles"]}),

        node_sets = {
            "atom": tfgnn.NodeSet.from_fields(
                sizes = mol_data["n_atoms"],
                features = {"atom_num": atom_nums,
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

def create_DFT_tensors(path):
    processDFTData(path)

    mol_df = pd.read_csv(path + "code/predicting_model/Shift/DFTNN/DFT_data_mol.csv.gz", index_col=0)
    atom_df = pd.read_csv(path + "code/predicting_model/Shift/DFTNN/DFT_data_atom.csv.gz", index_col=0)
    bond_df = pd.read_csv(path + "code/predicting_model/Shift/DFTNN/DFT_data_bond.csv.gz", index_col=0)

    train_data = tf.io.TFRecordWriter(path + "data/own_data/DFT_data_train.tfrecords")
    test_data = tf.io.TFRecordWriter(path + "data/own_data/DFT_data_test.tfrecords")
    valid_data = tf.io.TFRecordWriter(path + "data/own_data/DFT_data_valid.tfrecords")
    all_data = tf.io.TFRecordWriter(path + "data/own_data/all_DFT_data.tfrecords")

    total = len(mol_df)

    for idx, mol_id in tqdm(enumerate(mol_df["mol_id"])):
        mol_data = mol_df.loc[mol_df["mol_id"] == mol_id]
        atom_data = atom_df.loc[atom_df["mol_id"] == mol_id]
        bond_data = bond_df.loc[bond_df["mol_id"] == mol_id]

        if (mol_data['n_pro'].values == 0):
            continue
        
        # make sure the index resets to 0 instead of continuing from the previous molecule
        atom_data = atom_data.reset_index(drop=True)
        graph_tensor = create_graph_tensor_shift(mol_data, atom_data, bond_data)
        example = tfgnn.write_example(graph_tensor)
        
        all_data.write(example.SerializeToString())
        if idx < total*0.7:
            train_data.write(example.SerializeToString())
        elif idx < total*0.85:
            valid_data.write(example.SerializeToString())
        else:
            test_data.write(example.SerializeToString())


if __name__ == "__main__":
    #path = "/home1/s3665828/code/CASCADE/"
    path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    #path = "C:/Users/niels/Documents/repo/CASCADE/"

    create_DFT_tensors(path)

    '''
    # create dataframes if they do not exist yet
    if not os.path.isfile(path + "code/predicting_model/Shift/DFTNN/small_data_mol.csv.gz"):
        processData(path + 'data/own_data/small_dataset.txt', path)

    mol_df = pd.read_csv(path + "code/predicting_model/Shift/DFTNN/small_data_mol.csv.gz", index_col=0)
    atom_df = pd.read_csv(path + "code/predicting_model/Shift/DFTNN/small_data_atom.csv.gz", index_col=0)
    bond_df = pd.read_csv(path + "code/predicting_model/Shift/DFTNN/small_data_bond.csv.gz", index_col=0)

    mol_df = mol_df.sample(frac=1)  #shuffle

    train_data = tf.io.TFRecordWriter(path + "data/own_data/small_data_train.tfrecords")
    test_data = tf.io.TFRecordWriter(path + "data/own_data/small_data_test.tfrecords")
    valid_data = tf.io.TFRecordWriter(path + "data/own_data/small_data_valid.tfrecords")
    all_data = tf.io.TFRecordWriter(path + "data/own_data/all_small_data.tfrecords")

    total = len(mol_df)

    graph_schema = tfgnn.read_schema(path + "code/predicting_model/GraphSchemaMult.pbtxt")
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    for idx, mol_id in tqdm(enumerate(mol_df["mol_id"])):
        mol_data = mol_df.loc[mol_df["mol_id"] == mol_id]
        atom_data = atom_df.loc[atom_df["mol_id"] == mol_id]
        bond_data = bond_df.loc[bond_df["mol_id"] == mol_id]

        if (mol_data['n_pro'].values == 0):
            continue
        
        # make sure the index resets to 0 instead of continuing from the previous molecule
        atom_data = atom_data.reset_index(drop=True)
        graph_tensor = create_graph_tensor_shape(mol_data, atom_data, bond_data)
        example = tfgnn.write_example(graph_tensor)
        
        all_data.write(example.SerializeToString())
        if idx < total*0.7:
            train_data.write(example.SerializeToString())
        elif idx < total*0.85:
            valid_data.write(example.SerializeToString())
        else:
            test_data.write(example.SerializeToString())
'''