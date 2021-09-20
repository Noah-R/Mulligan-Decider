import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.models.load_model('model_9_19_2021')
data = pd.read_csv("preprocessed_data.csv", header=0, usecols=range(1, 13))
examples = data.iloc[:5]

features = {name: np.array(value) for name, value in examples.items()}
label = np.array(features.pop("won"))

preds = model.predict(x=features, verbose=1)

for i in range(len(preds)):
    print(examples.iloc[i, :])
    print("Predicted "+str(preds[i])+", result was "+str(label[i]))

#model.layers[1].get_weights()