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

from nfp.preprocessing import MolShapePreprocessor, GraphSequence, features

mols = pd.read_csv('code/predicting_model/Multiplicity/own_data_multiplicity_mol.csv.gz', index_col=0)
i = 0
print(mols)
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

    mols["Mol"][i] = new_mol
    i += 1

df = pd.read_csv('code/predicting_model/Multiplicity/own_data_multiplicity_atom.csv.gz', index_col=0)
#only choose C and H
df = df.loc[df.atom_type == 1]

#only predict peak shape for H of C-H
def to_C(atom):
    neighbors = [x.GetAtomicNum() for x in atom.GetNeighbors()]
    if 6 in neighbors: 
        return True
    else:
        return False
    
df['Mol'] = mols.reindex(df.mol_id).Mol.values
df = df.dropna()

df = df.loc[df.apply(lambda x: to_C(x['Mol'].GetAtomWithIdx(x['atom_index'])), axis=1).values]

grouped_df = df.groupby(['mol_id'])
df_Shape = []
shape_dict = {"mol_id":[], "Shape":[]}

'''for mol_id, df in grouped_df:
    for shape in df['Shape'].values:
        shape_matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        for i in range(6):
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
        
        shape_dict['mol_id'].append(mol_id)
        shape_dict['Shape'].append(np.asarray(shape_matrix).flatten())'''

for mol_id, df in grouped_df:
    for shape in df['Shape'].values:
        shape_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        for i in range(6):
                if (i >= len(shape)): shape_array[i*6 + 8] = 1
                elif (shape[i] == 'm'): shape_array[i*6 + 0] = 1
                elif (shape[i] == 's'): shape_array[i*6 + 1] = 1
                elif (shape[i] == 'd'): shape_array[i*6 + 2] = 1
                elif (shape[i] == 't'): shape_array[i*6 + 3] = 1
                elif (shape[i] == 'q'): shape_array[i*6 + 4] = 1
                elif (shape[i] == 'p'): shape_array[i*6 + 5] = 1
                elif (shape[i] == 'h'): shape_array[i*6 + 6] = 1
                elif (shape[i] == 'v'): shape_array[i*6 + 7] = 1
                else: print(shape[i])

                shape_dict['mol_id'].append(mol_id)
                shape_dict['Shape'].append(np.asarray(shape_array))
    
shape_df = pd.DataFrame.from_dict(shape_dict)

#print(shape_df["Shape"])
#print(shape_df["Shape"].values)
#print(shape_df["Shape"].dtype)

for mol_id, df in grouped_df:
    df_Shape.append([mol_id, df.Mol.values, df.atom_index.values.astype('int'), df.Shift.values.astype('double'), shape_df.loc[shape_df["mol_id"] == mol_id]["Shape"].values])
    #df_Shape.append([mol_id, df.atom_index.values.astype('int'), df.Shift.values.astype('double'), df.Shape.values])
    if len(df.atom_index.values) != len(set(df.atom_index.values)):
        print(mol_id)

df_Shape = pd.DataFrame(df_Shape, columns=['mol_id', 'Mol', 'atom_index', 'Shift', 'Shape'])

test = df_Shape.sample(n=10, random_state=666)
valid = df_Shape[~df_Shape.mol_id.isin(test.mol_id)].sample(n=10, random_state=666)
train = df_Shape[
    (~df_Shape.mol_id.isin(test.mol_id) & ~df_Shape.mol_id.isin(valid.mol_id))
              ]
test = test.set_index('mol_id')
valid = valid.set_index('mol_id')
train = train.set_index('mol_id')

test = mols.reindex(test.index).join(test[['atom_index', 'Shift', 'Shape']])
valid = mols.reindex(valid.index).join(valid[['atom_index', 'Shift', 'Shape']])
train = mols.reindex(train.index).join(train[['atom_index', 'Shift', 'Shape']])

test.to_pickle('code/predicting_model/Multiplicity/mult_test.pkl.gz', compression='gzip')
valid.to_pickle('code/predicting_model/Multiplicity/mult_valid.pkl.gz', compression='gzip')
train.to_pickle('code/predicting_model/Multiplicity/mult_train.pkl.gz', compression='gzip')

# Preprocess molecules
def Mol_iter(df):
    for index,r in df.iterrows():
        print(r)
        yield(r['Mol'], r['atom_index'], r['Shift'])

preprocessor = MolShapePreprocessor(
    n_neighbors=100, cutoff=5, atom_features=features.atom_features_shape)

#print(train)
inputs_train = preprocessor.fit(Mol_iter(train))
inputs_valid = preprocessor.predict(Mol_iter(valid))
inputs_test = preprocessor.predict(Mol_iter(test))

import pickle

with open('code/predicting_model/Multiplicity/mult_processed_inputs.p', 'wb') as file:        
    pickle.dump({
        'inputs_train': inputs_train,
        'inputs_valid': inputs_valid,
        'inputs_test': inputs_test,
        'preprocessor': preprocessor,
    }, file)
