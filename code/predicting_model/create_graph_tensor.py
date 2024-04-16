import pandas as pd
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

    mol_dict = {"mol_id":[], "smiles":[], "n_atoms":[]}
    atom_dict = {"mol_id":[], "atom_num":[], "Shift":[]}
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

        mol_dict["mol_id"].append(mol_id)
        mol_dict["smiles"].append(smiles)
        mol_dict["n_atoms"].append(n_atoms)
        
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
                atom_num = atom.GetAtomicNum()
                atom_dict["atom_num"].append(atom_num)
                
                if (atom_num == 1):
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


if __name__ == "__main__":
    # crate dataframes if they do not exist yet
    if not os.path.isfile("code/predicting_model/Shift/DFTNN/own_data_mol.csv.gz"):
        processData('data/own_data/cleaned_full_dataset.txt')

    mol_df = pd.read_csv("code/predicting_model/Shift/DFTNN/own_data_mol.csv.gz", index_col=0)
    atom_df = pd.read_csv("code/predicting_model/Shift/DFTNN/own_data_atom.csv.gz", index_col=0)
    bond_df = pd.read_csv("code/predicting_model/Shift/DFTNN/own_data_bond.csv.gz", index_col=0)
    
    for mol_id in mol_df["mol_id"]:
        for atom in atom_df.loc(atom_df["mol_id"] == mol_id):
            print(atom)