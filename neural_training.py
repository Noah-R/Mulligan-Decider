import numpy as np
import pandas as pd
import tensorflow as tf
from preprocessing import neuralPreprocess, trainTestSplit, getMulliganWinRates

#trainingdata, testdata = trainTestSplit(neuralPreprocess("game_data_public.STX.PremierDraft.csv"), .1)
#trainingdata.to_csv("training_data.csv")
#testdata.to_csv("test_data.csv")
testdata = pd.read_csv("test_data_29_sep.csv", header=0).drop(index=0, axis=1)
trainingdata = pd.read_csv("training_data_29_sep.csv", header=0).drop(index=0, axis=1)

target="won"
learningrate=.001
batchsize=1024
epochs=100
l2rate=.0001
dropoutrate=0.1
earlyStoppingPatience=10
date="30_sep_2021_1"

features=[]

for col in trainingdata.keys():
    if(col!=target and "Unnamed" not in col):
        features.append(tf.feature_column.numeric_column(col))

model = tf.keras.models.Sequential([
    tf.keras.layers.DenseFeatures(features),
    tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2rate), name='Hidden1'),
    tf.keras.layers.Dropout(rate=dropoutrate),
    tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2rate), name='Hidden2'),
    tf.keras.layers.Dropout(rate=dropoutrate),
    tf.keras.layers.Dense(units=1, activation=tf.sigmoid, name="Output")
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

tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir="tb_"+date, histogram_freq=1)
earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(patience=earlyStoppingPatience, verbose=1, restore_best_weights=True)

model.fit(
    x = features,
    y = label,
    batch_size = batchsize,
    epochs = epochs,
    shuffle = False,
    verbose = 2,
    validation_data = (testfeatures, testlabel), 
    callbacks=[tensorboardCallback, earlyStoppingCallback]
)

model.save("model_"+date)
getMulliganWinRates(trainingdata, 1)

#WARNING:tensorflow:Layers in a Sequential model should only have a single input tensor, but we receive a <class 'dict'>... Consider rewriting this model with the Functional API.
