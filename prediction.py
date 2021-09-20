import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.models.load_model('model_9_19_2021')
data = pd.read_csv("preprocessed_data.csv", header=0, usecols=range(1, 13))
examples = data.head(5)
features = {name: np.array(value) for name, value in examples.items()}
label = np.array(features.pop("won"))
preds = model.predict(x=features, verbose=1)
for i in range(len(preds)):
    print(examples.loc[i, :])
    print(preds[i])
print(model.layers[1].weights)
print(model.layers[1].bias.numpy())