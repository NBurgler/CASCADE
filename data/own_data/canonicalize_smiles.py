import fileinput
import rdkit
from rdkit import Chem

# Make the smiles canonical by converting them to molecules and back to smiles

with open('canon_dataset.txt', 'w') as new_file:
    old_file = open('output_all.txt', 'r')
    text = old_file.read()
    samples = text.split('\n\n')
    for sampleText in samples:
        sampleSplit = sampleText.split("\n")
        smiles = sampleSplit[0]
        sampleText = '\n'.join(sampleSplit[1:])
        new_file.write(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
        new_file.write('\n')
        new_file.write(sampleText)
        new_file.write('\n\n')