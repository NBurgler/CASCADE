#decanonize the molecules

with open('own_data_non_canon.txt', 'w') as decanonized_dataset:
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
