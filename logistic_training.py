import numpy as np
import pandas as pd
import tensorflow as tf
from preprocessing import logisticPreprocess, trainTestSplit

trainingdata, testdata = trainTestSplit(logisticPreprocess("game_data_public.STX.PremierDraft.csv"), .1)
trainingdata.to_csv("training_data.csv")
testdata.to_csv("test_data.csv")

target="won"
learningrate=.01
batchsize=32
epochs=256
date="25_sep_2021_1"

features=[]

for col in trainingdata.keys():
    if(col!=target and "Unnamed" not in col):
        features.append(tf.feature_column.numeric_column(col))

model = tf.keras.models.Sequential([
    tf.keras.layers.DenseFeatures(features),
    tf.keras.layers.Dense(units=1, input_shape=(1,) , activation=tf.sigmoid)
])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=learningrate),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["mse"]
)

features = {name: np.array(value) for name, value in trainingdata.items()}
features.pop("Unnamed: 0")
label = np.array(features.pop(target))

testfeatures = {name: np.array(value) for name, value in testdata.items()}
testlabel = np.array(testfeatures.pop(target))

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tb_"+date, histogram_freq=1)

model.fit(
    x = features,
    y = label,
    batch_size = batchsize,
    epochs = epochs,
    shuffle = False,
    verbose = 2,
    validation_data = (testfeatures, testlabel), 
    callbacks=[tensorboard_callback]
)

model.save("model_"+date)


#WARNING:tensorflow:Layers in a Sequential model should only have a single input tensor, but we receive a <class 'dict'>... Consider rewriting this model with the Functional API.