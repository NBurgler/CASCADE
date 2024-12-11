import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    #path = "/home1/s3665828/code/CASCADE/"
    #path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
    path = "C:/Users/niels/Documents/repo/CASCADE/"
    data = {"mol_id":[], "molecule":[], "atom_idx":[], "shift":[], "shape":[], "couplings":[]}

    atom_df = pd.read_csv(path + "code/predicting_model/Coupling/own_atom.csv.gz", index_col=0)
    dataset = atom_df.loc[atom_df["atom_symbol"] == "H"]
    dataset = dataset.drop(columns=["atom_symbol", "chiral_tag", "degree", "formal_charge", "hybridization", 
                            "is_aromatic", "no_implicit", "num_Hs", "valence"])
    dataset.to_csv(path + "code/matching_model/matching_data.csv.gz", compression='gzip')