import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

with open('attempts.txt', 'w') as record:
    bad_mols = open('bad_mols.txt', 'r')
    bad_mols = bad_mols.read()
    bad_mols = bad_mols.split('\n')
    for mol in bad_mols:
        if (mol == ""):
            continue
        mol = Chem.MolFromSmiles(mol)
        mol = Chem.AddHs(mol, addCoords=True)
        attempts = 0
        flag = 0
        while(flag == 0 and attempts <= 500):
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
            try:
                AllChem.MMFFOptimizeMolecule(mol)
                flag = 1
            except:
                attempts += 1
        
        mol = Chem.RemoveHs(mol)    #Save the mols without explicit Hydrogen atoms
        if flag == 0:
            print(f"{Chem.MolToSmiles(mol)} failed all {attempts} attempts.")
            record.write(f"{Chem.MolToSmiles(mol)}  FAILED \n")
        else:
            print(f"{Chem.MolToSmiles(mol)} succeeded after {attempts} attempts!")
            record.write(f"{Chem.MolToSmiles(mol)}  {attempts} \n")