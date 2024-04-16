with open('cleaned_full_dataset.txt', 'w') as new_file:
    old_file = open('full_dataset.txt', 'r')
    text = old_file.read()
    samples = text.split("\n\n")
    for sampleText in samples:
        sampleSplit = sampleText.split("\n")
        for line in sampleSplit:
            if (not line.startswith("CASE") and not line.startswith("ART")):
                new_file.write(line)
                new_file.write("\n")
            
        new_file.write("\n")