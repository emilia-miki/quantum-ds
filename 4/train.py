import rasterio
import geopandas as gpd
import tensorflow as tf
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import AffineTransformer
from shapely.geometry import Polygon
from rasterio import plot

# open the satellite imagery
with rasterio.open('T36UXV_20200406T083559_TCI_10m.jp2') as src:
    img = src.read()

# reshape it 
img = plot.reshape_as_image(img)

# open the shapefile
df = gpd.read_file('masks/Masks_T36UXV_20190427.shp')
df = df[df.geometry.notnull()] # remove null geometries
df = df.to_crs(src.crs) # convert to the CRS used by the .jp2 file

im_size = (src.meta['height'], src.meta['width'])

transformer = AffineTransformer(~src.transform)


# function to apply reverse AffineTransform from the .jp2 file
def apply_transform(polygon):
    poly_pts = []

    for i in polygon.exterior.coords:
        poly_pts.append(~src.transform * tuple(i))
    
    return Polygon(poly_pts)

# create a mask out of geometries in the geodataframe and apply the AffineTransform
mask = rasterize(shapes=df['geometry'].apply(apply_transform), out_shape=im_size)

# resize the mask
mask.resize(mask.shape[0], mask.shape[1], 1)


# function to randomly flip images horizontally for data augmentation
def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    return input_image, input_mask


# function to normalize image and mask
def normalize(input_image, input_mask):
    out_image = tf.cast(input_image, tf.float32) / 255.0
    # the mask is normalized by default
    return out_image, input_mask


# divide the data into training and testing datasets
dim = 128
thr = dim * 60

train_image = img[:thr, :, :]
train_mask = mask[:thr, :, :]

test_image = img[thr:, :, :]
test_mask = mask[thr:, :, :]

# normalize the data
train_image, train_mask = normalize(train_image, train_mask)
test_image, test_mask = normalize(test_image, test_mask)


# a generator-function to crop 512x512 images out of the main image
def gen(image, mask):
    h, w = image.shape[:2]
    h_pad = (dim - h % dim) % dim
    w_pad = (dim - w % dim) % dim
    image = np.pad(image, ((0, h_pad), (0, w_pad), (0, 0)), mode='constant')
    mask = np.pad(mask, ((0, h_pad), (0, w_pad), (0, 0)), mode='constant')

    for i in range(0, h_pad + h, dim):
        for j in range(0, w_pad + w, dim):
            yield image[i:i+dim, j:j+dim, :], mask[i:i+dim, j:j+dim, :]


# create the datasets
train_dataset = tf.data.Dataset.from_generator(lambda: gen(train_image, train_mask),
                                               output_signature=(
                                                   tf.TensorSpec(shape=(dim, dim, 3), dtype=train_image.dtype),
                                                   tf.TensorSpec(shape=(dim, dim, 1), dtype=train_mask.dtype)))

train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(lambda: gen(test_image, test_mask),
                                              output_signature=(
                                                  tf.TensorSpec(shape=(dim, dim, 3), dtype=test_image.dtype),
                                                  tf.TensorSpec(shape=(dim, dim, 1), dtype=test_mask.dtype)))

# technical specifications for the datasets
BATCH_SIZE = 32
BUFFER_SIZE = 1000
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_dataset = train_dataset.with_options(options)
train_batches = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_batches = test_dataset.take(2236).batch(BATCH_SIZE).with_options(options)
test_batches = test_dataset.skip(1864).take(372).batch(BATCH_SIZE).with_options(options)


# define the model
def double_conv_block(x, n_filters):
   x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

   return x


def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = tf.keras.layers.MaxPool2D(2)(f)
   p = tf.keras.layers.Dropout(0.3)(p)

   return f, p


def upsample_block(x, conv_features, n_filters):
    x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = tf.keras.layers.concatenate([x, conv_features])
    x = tf.keras.layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)

    return x


def build_unet_model():
    inputs = tf.keras.layers.Input(shape=(dim, dim, 3))

    # encoder: contracting path - downsample
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)

    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u9)

    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


# build the model and parallelize it
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    unet_model = build_unet_model()

# compile the model
unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=3.0),
                  metrics="accuracy")

# define some parameters and train the model
NUM_EPOCHS = 20

TRAIN_LENGTH = 5160
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

TEST_LENGTH = 2236
VALIDATION_STEPS = TEST_LENGTH // BATCH_SIZE

model_history = unet_model.fit(train_batches,
                               epochs=NUM_EPOCHS,
                               steps_per_epoch=STEPS_PER_EPOCH,
                               validation_steps=VALIDATION_STEPS,
                               validation_data=test_batches)

# export the model
unet_model.save('model.h5')
