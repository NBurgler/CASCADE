import pandas as pd

if __name__ == "__main__":
    path = "C:/Users/niels/Documents/repo/CASCADE/"
    atom_df = pd.read_csv(path + "code/predicting_model/Multiplicity/own_data_multiplicity_atom.csv.gz", index_col=0)
    print(atom_df["Shape"])