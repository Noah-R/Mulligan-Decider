import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def showExamples(num, data, preds, label):#only configured for neural-preprocessed data
    for i in range(num):
        d = data.iloc[np.random.randint(0, len(data.index)), :]
        d["on_play"] = d["on_play"]/7
        d["won"] = d["won"]/7
        d = d[d>0]*7
        print(d)
        print("Predicted "+str(preds[i])+", result was "+str(label[i]))

def plotCalibrationCurve(preds, label):
    prob_true, prob_pred = calibration_curve(label, preds, n_bins=100)
    plt.plot(prob_pred, prob_true)
    plt.plot([0, 1], [0, 1])
    plt.show()

def getAccuracy(preds, label, threshold=.5):
    correct=0
    for i in range(len(preds)):
        if(int(preds[i]>threshold)==label[i]):
            correct+=1
    print("Accuracy is "+str(correct/len(preds)))

def getWeights(model):#Only works for the logistic regression model, with features on layer 0 and output on layer 1.
    cols=model.layers[0].get_config()["feature_columns"]
    weights=model.layers[1].get_weights()[0]
    bias=model.layers[1].get_weights()[1]
    for index in range(len(cols)):
        print(str(cols[index]["config"]["key"])+": "+str(weights[index]))
    print("Bias: "+str(bias))

model = tf.keras.models.load_model('model_30_sep_2021_2')
data = pd.read_csv("test_data.csv", header=0)

features = {name: np.array(value) for name, value in data.items()}
label = np.array(features.pop("won"))

#preds = model.predict(x=features, verbose=1)
#showExamples(5, data, preds, label)
#plotCalibrationCurve(preds, label)
#getAccuracy(preds, label)
#model.evaluate(x=features, y=label, verbose=1)