import sys

sys.path.append('../..')

import pandas as pd 
import numpy as np

import pickle
import tqdm

from rdkit import Chem

file = open('../../../../data/own_data/cleaned_dataset.txt', 'r')
text = file.read()
samples = text.split('\n\n')
mol_id = 0

mol_dict = {"mol_id":[], "Mol":[], "n_atoms":[]}
atom_dict = {"mol_id":[], "atom_type":[], "atom_index":[], "Shift":[]}

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
        atom_dict["atom_index"].append(atom.GetIdx())#TODO check if needs +1
        
        if (atom_type == 1):
            atomSplit = sampleSplit[3+iter_H].split(",")
            atom_dict["Shift"].append(atomSplit[1])

            iter_H += 1
        else:
            atom_dict["Shift"].append("-")

    mol_id += 1

mol_df = pd.DataFrame.from_dict(mol_dict)
atom_df = pd.DataFrame.from_dict(atom_dict)

mol_df.to_csv("own_data_mol.csv.gz", compression='gzip')
atom_df.to_csv("own_data_atom.csv.gz", compression='gzip')