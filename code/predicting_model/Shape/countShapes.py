path =  "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/"
file = open(path + '/data/own_data/own_data_non_canon.txt', 'r')
text = file.read()
samples = text.split('\n\n')
count = {}

for sampleText in samples:
    sampleSplit = sampleText.split("\n")
    for line in sampleSplit[3:]:
        shape = line.split(",")[-1]
        if (shape == ''): print (sampleSplit[0])
        count[shape] = count.get(shape, 0) + 1

print(count)
