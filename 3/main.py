import numpy as np
import pandas as pd
import tensorflow as tf
import sys

# get the CSV filename
if len(sys.argv) == 1:
    filename = input('Enter the name of the CSV file with data: ')
elif len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    print('Too many arguments.')
    sys.exit(0)

# load data
data = np.genfromtxt(filename, 
                     delimiter=',', 
                     skip_header=1)

rows, cols = data.shape

# ignore all the columns except the 6th one
features = data[:, 6].reshape(rows, 1)

# normalize the data
mean = np.mean(features)
std = np.std(features)
features = (features - mean) / std

# add the squares of the data
features = np.hstack([features, np.square(features)])

# load the model and get predictions
model = tf.keras.models.load_model('model.h5')

predictions = model.predict(features).reshape(rows)

# read mean and std of the targets from 'targets_specs.txt'
with open('targets_specs.txt') as f:
    line = f.readline()

tokens = line.split(' ')
mean = float(tokens[0])
std = float(tokens[1])

# reverse normalization of the targets
predictions = predictions * std + mean

# append the resulting predictions to the CSV file
csv = pd.read_csv(filename)
csv['target'] = predictions
csv.to_csv(filename, index=False)
