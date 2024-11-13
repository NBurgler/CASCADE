import pandas as pd

if __name__ == "__main__":
    path = "C:/Users/niels/Documents/repo/CASCADE/"
    atom_df = pd.read_csv(path + "code/predicting_model/Shape/own_atom.csv.gz", index_col=0)
    H_atom_df = atom_df.loc[atom_df["atom_symbol"] == "H"]
    count = {"first_token":[], "second_token":[], "third_token":[], "fourth_token":[], "fifth_token":[], "sixth_token":[]}
    for shape in H_atom_df["Shape"]:
        i = 0
        for token in shape:
            if i == 0: count["first_token"].append(token)
            elif i == 1: count["second_token"].append(token)
            elif i == 2: count["third_token"].append(token)
            elif i == 3: count["fourth_token"].append(token)
            elif i == 4: count["fifth_token"].append(token)
            elif i == 5: count["sixth_token"].append(token)
            i += 1

        while i != 6:
            if i == 0: count["first_token"].append("s")
            elif i == 1: count["second_token"].append("s")
            elif i == 2: count["third_token"].append("s")
            elif i == 3: count["fourth_token"].append("s")
            elif i == 4: count["fifth_token"].append("s")
            elif i == 5: count["sixth_token"].append("s")
            i += 1

    count_df = pd.DataFrame.from_dict(count)
    count_df = count_df.apply(pd.value_counts).fillna(0).astype("int64")
    count_df = count_df.reindex(["m", "s", "d", 't', "q", "p", "h", "v"])
    print(count_df)
    total = count_df["first_token"].sum()
    print("First token S percentage: " + str(round((count_df.at["s", "first_token"] / total) * 100, 2)) + "%")
    print("Second token S percentage: " + str(round((count_df.at["s", "second_token"] / total) * 100, 2)) + "%")
    print("Third token S percentage: " + str(round((count_df.at["s", "third_token"] / total) * 100, 2)) + "%")
    print("Fourth token S percentage: " + str(round((count_df.at["s", "fourth_token"] / total) * 100, 2)) + "%")
    print("Fifth token S percentage: " + str(round((count_df.at["s", "fifth_token"] / total) * 100, 2)) + "%")
    print("Sixth token S percentage: " + str(round((count_df.at["s", "sixth_token"] / total) * 100, 2)) + "%")