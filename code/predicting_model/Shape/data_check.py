import sys
sys.path.append('../..')
import pickle
from nfp.preprocessing import MolPreprocessor, GraphSequence
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

def atomic_number_tokenizer(atom):
    return atom.GetNumRadicalElectrons()

with open('own_processed_inputs.p', 'rb') as f:
    input_data = pickle.load(f)

print(input_data)