# Remove the "bad_mols" from the dataset

with open('cleaned_dataset.txt', 'w') as cleaned_dataset:
    dataset = open('canon_dataset.txt', 'r')
    bad_mols = open('bad_mols.txt', 'r')
    bad_mols = bad_mols.read().split('\n')
    text = dataset.read()
    samples = text.split('\n\n')
    for sampleText in samples:
        sampleSplit = sampleText.split("\n")
        smiles = sampleSplit[0]
        if smiles not in bad_mols:
            cleaned_dataset.write(sampleText)
            cleaned_dataset.write('\n\n')