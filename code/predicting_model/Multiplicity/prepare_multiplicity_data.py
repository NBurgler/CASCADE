import sys

import pandas as pd 
import numpy as np

import pickle

from rdkit import Chem

file = open('data/own_data/small_dataset.txt', 'r')
text = file.read()
samples = text.split('\n\n')
mol_id = 0

mol_dict = {"mol_id":[], "Mol":[], "n_atoms":[]}
atom_dict = {"mol_id":[], "atom_type":[], "atom_index":[], "Shift":[], "Shape":[]}

for sampleText in samples:
    sampleSplit = sampleText.split("\n")
    smiles = sampleSplit[0]

    if (smiles == ""):
        continue
    
    smilesH = sampleSplit[1]

    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms(onlyExplicit=False)

    mol_dict["mol_id"].append(mol_id)
    mol_dict["Mol"].append(smiles)
    mol_dict["n_atoms"].append(n_atoms)

    iter_H = 0
    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
        atom_dict["mol_id"].append(mol_id)
        atom_type = atom.GetAtomicNum()
        atom_dict["atom_type"].append(atom_type)
        atom_dict["atom_index"].append(atom.GetIdx())
                       # m  s  d  t  q  p  h hept -
        shape_matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]]

        if (atom_type == 1):
            atomSplit = sampleSplit[3+iter_H].split(",")
            atom_dict["Shift"].append(float(atomSplit[1]))
            shape = atomSplit[3]
            '''for i in range(6):
                if (i >= len(shape)): shape_matrix[i][8] = 1
                elif (shape[i] == 'm'): shape_matrix[i][0] = 1
                elif (shape[i] == 's'): shape_matrix[i][1] = 1
                elif (shape[i] == 'd'): shape_matrix[i][2] = 1
                elif (shape[i] == 't'): shape_matrix[i][3] = 1
                elif (shape[i] == 'q'): shape_matrix[i][4] = 1
                elif (shape[i] == 'p'): shape_matrix[i][5] = 1
                elif (shape[i] == 'h'): shape_matrix[i][6] = 1
                elif (shape[i] == 'v'): shape_matrix[i][7] = 1
                else: print(shape[i])
                
            atom_dict["Shape"].append(shape_matrix)'''
            atom_dict["Shape"].append(shape)

            iter_H += 1
        else:
            atom_dict["Shift"].append("-")
            atom_dict["Shape"].append("-")
        
    mol_id += 1

mol_df = pd.DataFrame.from_dict(mol_dict)
atom_df = pd.DataFrame.from_dict(atom_dict)

mol_df.to_csv("code/predicting_model/Multiplicity/own_data_multiplicity_mol.csv.gz", compression='gzip')
atom_df.to_csv("code/predicting_model/Multiplicity/own_data_multiplicity_atom.csv.gz", compression='gzip')
