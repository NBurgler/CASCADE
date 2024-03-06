import pandas as pd 
import pickle
import rdkit
from rdkit import Chem

data = pd.read_pickle('own_train.pkl.gz')
i = 0
for mol in data['Mol']:
    data['Mol'][i] = Chem.MolToSmiles(mol)
    i += 1
data.to_csv("check.txt")