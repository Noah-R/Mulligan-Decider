import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def showExamples(num, data, preds, label):
    for i in range(num):
        print(data.iloc[i, :])
        print("Predicted "+str(preds[i])+", result was "+str(label[i]))

def plotCalibrationCurve(preds, label):
    prob_true, prob_pred = calibration_curve(label, preds, n_bins=100)
    plt.plot(prob_pred, prob_true)
    plt.plot([0, 1], [0, 1])
    plt.show()

def getWeights(model):#Only works for the logistic regression model, with features on layer 0 and output on layer 1.
    cols=model.layers[0].get_config()["feature_columns"]
    weights=model.layers[1].get_weights()[0]
    bias=model.layers[1].get_weights()[1]
    for index in range(len(cols)):
        print(str(cols[index]["config"]["key"])+": "+str(weights[index]))
    print("Bias: "+str(bias))

model = tf.keras.models.load_model('archives/model_21_sep_2021_1')
data = pd.read_csv("test_data.csv", header=0)
data = data.drop(index=0, axis=1)

features = {name: np.array(value) for name, value in data.items()}
label = np.array(features.pop("won"))
preds = model.predict(x=features, verbose=1)

showExamples(5, data, preds, label)
plotCalibrationCurve(preds, label)
getWeights(model)

"""
cd desktop/mulligan-decider
tensorboard --logdir tb_modelname
"""