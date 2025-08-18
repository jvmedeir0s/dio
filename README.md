# dio
projetos da dio
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

import os
import zipfile
import urllib.request

url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
file_path = 'cats_and_dogs_filtered.zip'

urllib.request.urlretrieve(url, file_path)

with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall('data/')

base_dir = 'data/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

from tensorflow.keras.preprocessing import image_dataset_from_directory

train_dataset = image_dataset_from_directory(train_dir,
                                             image_size=(150, 150),
                                             batch_size=32)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  image_size=(150, 150),
                                                  batch_size=32)





base_model = keras.applications.MobileNetV2(
    input_shape=(150, 150, 3),
    include_top=False,
    weights='imagenet'
)


base_model.trainable = False


inputs = keras.Input(shape=(150, 150, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset
)

test_loss, test_acc = model.evaluate(validation_dataset)
print(f'\nAcurácia no conjunto de validação: {test_acc}')
