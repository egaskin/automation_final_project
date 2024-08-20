import numpy as np
import csv

# Open the CSV file
data = np.zeros((454, 32), dtype=float)
with open('data/inhibition_data/inhibition_features.csv', newline='') as file:
    reader = csv.reader(file)
    next(reader)

    # Iterate over each row in the CSV file
    for i, row in enumerate(reader):
        data[i] = np.array(row)

y = data[:, 2]
X = data[:, 3:]

np.save('data/inhibition_data/X.npy', X)
np.save('data/inhibition_data/y.npy', y)