import numpy as np
import pandas as pd
import tensorflow as tf
from preprocessing import getMulliganWinRates
from itertools import combinations

#configured for neural network
#example is a list, list[0] is number of cards as a float, list[1] is play/draw as 1.0 or 0.0, list [2:] are the names of cards in hand as formattedon front of card
def predictExample(example, model, keys):
    if(len(example)==example[0]+2):
        features = {name: np.array([0.0]) for name in keys}
        features["cards"][0]=example[0]/7
        features["on_play"][0]=example[1]
        for card in example[2:]:
            features["opening_hand_"+card.replace(" ", "_").replace(",", "").replace("'", "")][0]+=1/7
        preds = model.predict(x=features, verbose=0)
        return ("Probability of winning: "+str(preds[0]))
    
    elif(len(example)>example[0]+2):#this could also do what the above does, but it's a little slower
        possibleHands = list(combinations(range(2, len(example)), int(example[0])))
        features = {name: np.full(shape=len(possibleHands), fill_value=0.0, dtype=np.float) for name in keys}
        for i in range(len(possibleHands)):
            features["cards"][i]=example[0]/7
            features["on_play"][i]=example[1]
            indexes=list(possibleHands[i])
            for index in indexes:
                card=example[index]
                features["opening_hand_"+card.replace(" ", "_").replace(",", "").replace("'", "")][i]+=1/7
        preds = model.predict(x=features, verbose=0)
        #for i in range(len(possibleHands)):#remove this for loop to only show the best hand
        #    handIndices = list(possibleHands[i])
        #    hand = []
        #    for index in handIndices:
        #        hand.append(example[index])
        #    print("This hand: "+str(hand)+" has this probability of winning: "+str(preds[i]))
        bestIndex = np.argmax(preds)
        bestHandIndices = list(possibleHands[bestIndex])
        bestHand = []
        for index in bestHandIndices:
            bestHand.append(example[index])
        return ("Keep the following cards: "+str(bestHand)+"\nProbability of winning: "+str(preds[bestIndex]))
    
    else:
        print("Something went seriously wrong here, the hand is missing some cards")

def enterExample():
    hand = [float(input("Number of cards")), float(input("On play? 1 for yes, 0 for no."))]
    count = int(input("Number of cards to enter"))
    for i in range(count):
        hand.append(input("Enter card name"))
    return hand

def setup(modelname, keyfile):
    model = tf.keras.models.load_model(modelname)
    keys=open(keyfile, "r").read().strip('][').strip('\"').split('\", \"')
    return model, keys