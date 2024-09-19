import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import gzip

if __name__ == "__main__":
    #path = "C:/Users/niels/Documents/repo/CASCADE/"
    path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    
    shift_df = pd.read_csv(path + "data/DFT8K/DFT8K.csv.gz", index_col=0)
    print(shift_df)
    mol_id = 20176935
    print(shift_df.loc[(shift_df['mol_id'] == mol_id) & (shift_df['atom_index'] == 2)]["Shift"])

    with gzip.open(path + "data/DFT8K/DFT.sdf.gz", 'rb') as dft:
        mol_suppl = Chem.ForwardSDMolSupplier(dft, removeHs=False)
        #for mol in mol_suppl:
            #print(Chem.MolToSmiles(mol))
            #print(mol.GetProp("_Name"))
            #print(Chem.Get3DDistanceMatrix(mol))
            #for atom in mol.GetAtoms():
                #print(atom.GetIdx())
                #print(atom.GetSymbol())


    #print(shift_df.loc[shift_df['mol_id'] == 11224])
