import numpy as np
import pandas as pd
import tensorflow as tf
from preprocessing import getMulliganWinRates
from itertools import combinations

def enterExample():
    hand = [float(input("Number of cards")), float(input("On play? 1 for yes, 0 for no."))]
    count = int(input("Number of cards to enter"))
    for i in range(count):
        hand.append(input("Enter card name"))
    return hand

def predictExample(example, model, features):#configured for neural network
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


#data = pd.read_csv("training_data.csv", header=0)
#predictExample(enterExample(), model, features)
#getMulliganWinRates(data, 1)