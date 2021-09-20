import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from preprocessing import preprocess

data = preprocess("game_data_public.STX.PremierDraft.csv")
data.to_csv("preprocessed_data.csv")

target="won"
learningrate=.01
batchsize=10
epochs=25

features=[]

for col in data.keys():
    if(col!=target):
        features.append(tf.feature_column.numeric_column(col))

model = tf.keras.models.Sequential([
    layers.DenseFeatures(features),
    layers.Dense(units=1, input_shape=(1,) , activation=tf.sigmoid)
])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=learningrate),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["mae"]
)

features = {name: np.array(value) for name, value in data.items()}
label = np.array(features.pop(target)) 

model.fit(
    x = features,
    y = label,
    batch_size = batchsize,
    epochs = epochs,
    shuffle = True,
    verbose = 2,
    validation_split = 0.1
)

model.save("model_9_19_2021")


#WARNING:tensorflow:Layers in a Sequential model should only have a single input tensor, but we receive a <class 'dict'>... Consider rewriting this model with the Functional API.