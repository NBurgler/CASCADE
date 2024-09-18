import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import gzip

if __name__ == "__main__":
    path = "C:/Users/niels/Documents/repo/CASCADE/"
    shift_df = pd.read_csv(path + "data/DFT8K/DFT8K.csv.gz", index_col=0)
    print(shift_df)

    with gzip.open(path + "data/DFT8K/DFT.sdf.gz", 'rb') as dft:
        text = dft.read()
        textSplit = text.split("\n")
        print(textSplit[0])
        mol_suppl = Chem.ForwardSDMolSupplier(dft)
        for mol in mol_suppl:
            print(Chem.MolToSmiles(mol))
