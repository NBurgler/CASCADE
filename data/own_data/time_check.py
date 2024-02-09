import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from time import process_time

mol = Chem.MolFromSmiles("C1=C2COC13COC2C3")
#mol = Chem.AddHs(mol, addCoords=True)
attempts = 0
flag = 0
t1_start = process_time()
while(flag == 0 and attempts <= 500):
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
            try:
                AllChem.MMFFOptimizeMolecule(mol)
                flag = 1
            except:
                attempts += 1

print(attempts)
print(Chem.MolToMolBlock(mol))
t1_stop = process_time()

print("Elapsed time:", t1_stop, t1_start)
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)