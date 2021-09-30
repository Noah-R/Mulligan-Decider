import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from preprocessing import getMulliganWinRates

def showExamples(num, data, preds, label):#only configured for neural-preprocessed data
    for i in range(num):
        d = data.iloc[i, :]
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

def enterExample(model, data):#configured for neural network
    features = {name: np.array([0.0]) for name, value in data.items()}
    features.pop("Unnamed: 0")
    features.pop("won")
    cards = float(input("Number of cards"))
    onplay = float(input("On play? 1 for yes, 0 for no."))
    features["cards"][0]=cards/7
    features["on_play"][0]=onplay
    for i in range(int(cards)):
        card = input("Enter card name").replace(" ", "_").replace(",", "").replace("'", "")
        features["opening_hand_"+card][0]+=1/7
    preds = model.predict(x=features, verbose=1)
    print(preds[0])


model = tf.keras.models.load_model('model_30_sep_2021_1')
data = pd.read_csv("test_data.csv", header=0)

features = {name: np.array(value) for name, value in data.items()}
label = np.array(features.pop("won"))

preds = model.predict(x=features, verbose=1)
showExamples(20, data, preds, label)
plotCalibrationCurve(preds, label)
getAccuracy(preds, label)
#model.evaluate(x=features, y=label, verbose=1)

#data = pd.read_csv("training_data.csv", header=0).drop(index=0, axis=1)
#getMulliganWinRates(data, 1)
#enterExample(model, data)