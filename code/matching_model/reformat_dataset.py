import pandas as pd

if __name__ == "__main__":
    path = "C:/Users/niels/Documents/repo/CASCADE/"
    mol_df = pd.read_csv(path + "code/predicting_model/All/own_mol.csv.gz", index_col=0)
    atom_df = pd.read_csv(path + "code/predicting_model/all/own_atom.csv.gz", index_col=0)
    atom_df = atom_df.loc[atom_df["atom_symbol"] == "H", ["mol_id", "atom_idx", "Shift", "Shape", "Coupling"]]
    peak_list = []

    for i, mol in mol_df.iterrows():
        atoms = atom_df.loc[atom_df["mol_id"] == mol["mol_id"]]
        atoms = atoms.sort_values(by="Shift", ascending=False)
        atoms["Group"] = (atoms["Shift"].diff() > 0.1).cumsum()
        atoms["Peak"] = atoms.groupby(["Group", "Shape", "Coupling"]).ngroup()
        peaks = atoms.groupby(["mol_id", "Peak", "Shift", "Shape", "Coupling"])['atom_idx'].apply(list).reset_index()
        peaks = peaks.sort_values(by="Shift", ascending=False)
        peaks = peaks.drop("Peak", axis=1)
        for j, peak in peaks.iterrows():
            peak_data = {"mol_id": [], "Shift": [], "Shape": [], "Coupling": [], "atom_idx": []}
            peak_data["mol_id"] = peak["mol_id"]
            peak_data["Shift"] = peak["Shift"]
            peak_data["Shape"] = peak["Shape"]
            peak_data["Coupling"] = peak["Coupling"]
            peak_data["atom_idx"] = peak["atom_idx"]
            peak_list.append(peak_data)

    peak_df = pd.DataFrame(peak_list)
    #peak_df.to_csv(path + "code/matching_model/peaks.csv.gz", compression='gzip')
    print(peak_df)