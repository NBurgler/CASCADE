import numpy as np
import matplotlib.pyplot as plt


with open('data/own_data/attempts.txt') as file:
    attempts = file.read().split('\n')
    count = np.array(0, dtype=float)
    for attempt in attempts:
        if(attempt == ''): continue
        value = attempt.split()[1]
        if value == "FAILED":
            count = np.append(count, float(500))
        else:
            count = np.append(count, float(value))
    print(count)
    #count, bins = np.histogram(count)
    plt.hist(count, bins=20)
    plt.show()