import sys

#change the path into where the nfp folder is
sys.path.append('code/predicting_model')

import pandas as pd
import numpy as np
import gzip

import warnings
from tqdm import tqdm

import gzip
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ForwardSDMolSupplier

from itertools import islice

from nfp.preprocessing import MolAPreprocessor, GraphSequence

path = "C:/Users/niels/Documents/repo/CASCADE/"
dataset = "cascade"

if (dataset == "cascade"):
    mols = []
    with gzip.open(path + 'data/DFT8K/DFT.sdf.gz', 'rb') as sdfile:
        mol_supplier = ForwardSDMolSupplier(sdfile, removeHs=False)
        for mol in tqdm(mol_supplier):
            if mol:
                mols += [(int(mol.GetProp('_Name')), mol, mol.GetNumAtoms())]

    mols = pd.DataFrame(mols, columns=['mol_id', 'Mol', 'n_atoms'])
    mols = mols.set_index('mol_id', drop=True)

    df = pd.read_csv(path + 'data/DFT8K/DFT8K.csv.gz', index_col=0)
    #only choose H
    df = df.loc[df['atom_type'] == 1]
    print(df)

elif (dataset == "own"):
    mols = pd.read_csv('code/predicting_model/Shift/DFTNN/own_data_mol.csv.gz', index_col=0)
    i = 0
    for mol in mols["Mol"]:
        new_mol = Chem.MolFromSmiles(mol)
        new_mol = Chem.AddHs(new_mol, addCoords=True)
        attempts = 0
        flag = 0

        while(flag == 0 and attempts <= 500):   # Attempt to find a proper embedding 500 times
            AllChem.EmbedMolecule(new_mol, useRandomCoords=True)
            try:
                AllChem.MMFFOptimizeMolecule(new_mol)
                flag = 1
            except:
                attempts += 1

        if (flag == 0):
            bad_mol = Chem.RemoveHs(new_mol)
            print("Error with optimizing")
            print(Chem.MolToSmiles(bad_mol))
            print(attempts)
            continue

        try:
            Chem.rdMolTransforms.CanonicalizeMol(new_mol, normalizeCovar=True, ignoreHs=False)
        except ValueError:
            bad_mol = Chem.RemoveHs(new_mol)
            print("Error with canonicalizing")
            print(Chem.MolToSmiles(bad_mol))
            continue

        mols.loc[i, "Mol"] = new_mol
        i += 1

    df = pd.read_csv('code/predicting_model/Shift/DFTNN/own_data_atom.csv.gz', index_col=0)
    #only choose C and H
    df = df.loc[df.atom_type == 1]

#only predict chemical shift for H of C-H
def to_C(atom):
    neighbors = [x.GetAtomicNum() for x in atom.GetNeighbors()]
    if 6 in neighbors: 
        return True
    else:
        return False

df['Mol'] = mols.reindex(df['mol_id'])['Mol'].values
df = df.dropna()
df = df.loc[df.apply(lambda x: to_C(x['Mol'].GetAtomWithIdx(x['atom_index'])), axis=1).values]

grouped_df = df.groupby(['mol_id'])
df_Shift = []
for mol_id,df in grouped_df:
    df_Shift.append([mol_id[0], df.atom_index.values.astype('int'), df.Shift.values.astype(np.float32)])
    if len(df.atom_index.values) != len(set(df.atom_index.values)):
        print(mol_id)

df_Shift = pd.DataFrame(df_Shift, columns=['mol_id', 'atom_index', 'Shift']) 

test = df_Shift.sample(n=500, random_state=666)
valid = df_Shift[~df_Shift.mol_id.isin(test.mol_id)].sample(n=500, random_state=666)
train = df_Shift[
    (~df_Shift.mol_id.isin(test.mol_id) & ~df_Shift.mol_id.isin(valid.mol_id))
              ]
test = test.set_index('mol_id')
valid = valid.set_index('mol_id')
train = train.set_index('mol_id')

test = mols.reindex(test.index).join(test[['atom_index', 'Shift']])
valid = mols.reindex(valid.index).join(valid[['atom_index', 'Shift']])
train = mols.reindex(train.index).join(train[['atom_index', 'Shift']])

if (dataset == "cascade"):
    test.to_pickle(path + 'code/predicting_model/Shift/DFTNN/cascade_test.pkl.gz', compression='gzip')
    valid.to_pickle(path + 'code/predicting_model/Shift/DFTNN/cascade_valid.pkl.gz', compression='gzip')
    train.to_pickle(path + 'code/predicting_model/Shift/DFTNN/cascade_train.pkl.gz', compression='gzip')
elif (dataset == "own"):
    test.to_pickle('code/predicting_model/Shift/DFTNN/own_test.pkl.gz', compression='gzip')
    valid.to_pickle('code/predicting_model/Shift/DFTNN/own_valid.pkl.gz', compression='gzip')
    train.to_pickle('code/predicting_model/Shift/DFTNN/own_train.pkl.gz', compression='gzip')

# Preprocess molecules
def atomic_number_tokenizer(atom):
    return atom.GetAtomicNum()
def Mol_iter(df):
    for index,r in df.iterrows():
        yield(r['Mol'], r['atom_index'])

preprocessor = MolAPreprocessor(
    n_neighbors=100, cutoff=5, atom_features=atomic_number_tokenizer)

inputs_train = preprocessor.fit(Mol_iter(train))
inputs_valid = preprocessor.predict(Mol_iter(valid))
inputs_test = preprocessor.predict(Mol_iter(test))

import pickle

if (dataset == "cascade"):  
    with open(path + 'code/predicting_model/Shift/DFTNN/cascade_processed_inputs.p', 'wb') as file:        
        pickle.dump({
            'inputs_train': inputs_train,
            'inputs_valid': inputs_valid,
            'inputs_test': inputs_test,
            'preprocessor': preprocessor,
        }, file)
elif (dataset == "own"):  
    with open('code/predicting_model/Shift/DFTNN/own_processed_inputs.p', 'wb') as file:        
        pickle.dump({
            'inputs_train': inputs_train,
            'inputs_valid': inputs_valid,
            'inputs_test': inputs_test,
            'preprocessor': preprocessor,
        }, file)
