import numpy as np
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

# load the data
data = np.genfromtxt(filename, 
                     delimiter=',', 
                     skip_header=1)

rows, cols = data.shape

features = data[:, :cols - 1]
targets = data[:, cols - 1]

# drop all the feature columns except the 6th one
features = features[:, 6].reshape(rows, 1)

# normalize the data
mean = np.mean(features)
std = np.std(features)
features = (features - mean) / std

mean = np.mean(targets)
std = np.std(targets)
targets = (targets - mean) / std

# add the squares of the feature
features = np.hstack([features, np.square(features)])

# create a parallelized model
learning_rate = 0.01

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(units=1)
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredError()
    metric = tf.keras.metrics.RootMeanSquaredError()

model.compile(optimizer=optimizer,
            loss=loss,
            metrics=[metric])

# create a dataset and change auto_shard_policy to DATA
ds = tf.data.Dataset.from_tensor_slices((features, targets)).shuffle(rows).batch(256)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
ds = ds.with_options(options)

# train the model
model.fit(ds, epochs=5)

# export the model
model.save('model.h5')
with open('targets_specs.txt', 'w') as f:
    f.write(f'{mean} {std}')
