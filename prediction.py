import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('model_20_sep_2021_2')
data = pd.read_csv("preprocessed_data.csv", header=0, usecols=range(1, 13))
examples = data.iloc[:100000]

features = {name: np.array(value) for name, value in examples.items()}
label = np.array(features.pop("won"))
preds = model.predict(x=features, verbose=1)

"""
for i in range(len(preds)):
    print(examples.iloc[i, :])
    print("Predicted "+str(preds[i])+", result was "+str(label[i]))
"""

prob_true, prob_pred = calibration_curve(label, preds, n_bins=100)
plt.plot(prob_pred, prob_true)
plt.show()
print(prob_pred)
print(prob_true)

#print(model.layers[0].get_config()["feature_columns"])
#print(model.layers[1].get_weights())

"""
cd desktop/mulligan-decider
tensorboard --logdir tb_20_sep_2021_2
"""