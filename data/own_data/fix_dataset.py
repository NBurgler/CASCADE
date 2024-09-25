
#decanonize the molecules
def decanonize_mols(path):
    with open(path + 'own_data_non_canon.txt', 'w') as decanonized_dataset:
        canon_dataset = open('canon_full_dataset.txt', 'r')
        non_canon_dataset = open('output_all.txt', 'r')

        canon_text = canon_dataset.read()
        canon_samples = canon_text.split('\n\n')

        non_canon_text = non_canon_dataset.read()
        non_canon_samples = non_canon_text.split("\n")
        for sampleText in canon_samples:
            sampleSplit = sampleText.split("\n")
            smiles = sampleSplit[0]
            smiles_with_H = sampleSplit[1]
            index = non_canon_samples.index(smiles_with_H)
            new_smiles = non_canon_samples[index-1]
            decanonized_dataset.write(new_smiles)
            decanonized_dataset.write("\n")
            decanonized_dataset.write("\n".join(sampleSplit[1:]))
            decanonized_dataset.write("\n")
            decanonized_dataset.write("\n")

def remove_negative(path):
    with open(path + 'updated_data_non_canon.txt', 'w') as new_dataset:
        non_canon_dataset = open(path + 'own_data_non_canon.txt', 'r')
        non_canon_text = non_canon_dataset.read()
        non_canon_samples = non_canon_text.split("\n\n")
        for sampleText in non_canon_samples:
            save = 1
            sampleLines = sampleText.split("\n")
            for line in sampleLines[3:]:
                if (line == "NO MULTIPLET ASSIGNED"):
                    save = 0
                    continue
        
                shift = line.split(',')[1]

                if (float(shift) < 0.0):
                    save = 0

            if save:
                new_dataset.write(sampleText)
                new_dataset.write("\n\n")

def add_mol_id(path):
    with open(path + 'own_data_with_id.txt', 'w') as new_dataset:
        non_canon_dataset = open(path + 'updated_data_non_canon.txt', 'r')
        non_canon_text = non_canon_dataset.read()
        non_canon_samples = non_canon_text.split("\n\n")
        mol_id = 0
        for sampleText in non_canon_samples:
            new_dataset.write(str(mol_id) + "\n")
            new_dataset.write(sampleText)
            new_dataset.write("\n\n")
            mol_id += 1


if __name__ == "__main__":
    path = "C:/Users/niels/Documents/repo/CASCADE/data/own_data/"
    add_mol_id(path)