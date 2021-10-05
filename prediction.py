import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from preprocessing import getMulliganWinRates
from itertools import combinations

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

def enterExample():
    hand = [float(input("Number of cards")), float(input("On play? 1 for yes, 0 for no."))]
    count = int(input("Number of cards to enter"))
    for i in range(count):
        hand.append(input("Enter card name"))
    return hand

def predictExample(example, model, features):#configured for neural network, should reconfigure for features parameter instead of data
    if(len(example)==example[0]+2):
        features = {name: np.array([0.0]) for name in features.keys()}
        features["cards"][0]=example[0]/7
        features["on_play"][0]=example[1]
        for card in example[2:]:
            features["opening_hand_"+card.replace(" ", "_").replace(",", "").replace("'", "")][0]+=1/7
        preds = model.predict(x=features, verbose=1)
        print("Probability of winning: "+str(preds[0]))
    
    elif(len(example)>example[0]+2):#this could also do what the above does, but it's a little slower
        possibleHands = list(combinations(range(2, len(example)), int(example[0])))
        features = {name: np.full(shape=len(possibleHands), fill_value=0.0, dtype=np.float) for name in features.keys()}
        for i in range(len(possibleHands)):
            features["cards"][i]=example[0]/7
            features["on_play"][i]=example[1]
            indexes=list(possibleHands[i])
            for index in indexes:
                card=example[index]
                features["opening_hand_"+card.replace(" ", "_").replace(",", "").replace("'", "")][i]+=1/7
        preds = model.predict(x=features, verbose=1)
        for i in range(len(possibleHands)):#remove this for loop to only show the best hand
            handIndices = list(possibleHands[i])
            hand = []
            for index in handIndices:
                hand.append(example[index])
            print("This hand: "+str(hand)+" has this probability of winning: "+str(preds[i]))
        bestIndex = np.argmax(preds)
        bestHandIndices = list(possibleHands[bestIndex])
        bestHand = []
        for index in bestHandIndices:
            bestHand.append(example[index])
        print("Keep the following cards: "+str(bestHand))
        print("Probability of winning: "+str(preds[bestIndex]))
    
    else:
        print("Something went seriously wrong here, the hand is missing some cards")


model = tf.keras.models.load_model('model_30_sep_2021_2')
data = pd.read_csv("test_data.csv", header=0)

features = {name: np.array(value) for name, value in data.items()}
label = np.array(features.pop("won"))

#preds = model.predict(x=features, verbose=1)
#showExamples(5, data, preds, label)
#plotCalibrationCurve(preds, label)
#getAccuracy(preds, label)
#model.evaluate(x=features, y=label, verbose=1)

#data = pd.read_csv("training_data.csv", header=0)
#getMulliganWinRates(data, 1)
predictExample(enterExample(), model, features)