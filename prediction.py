import pandas as pd
from preprocessing import getMulliganWinRates
from itertools import combinations
import requests
import json

#configured for neural network, example is a list, list[0] is number of cards as a float, list[1] is play/draw as 1.0 or 0.0, list [2:] are the names of cards in hand as formatted on front of card, a dict would have been better practice, but it already works, so no sense doing extra work to make it less efficient
def predictExample(example, keys, mulliganWinRates):
    if(len(example)==example[0]+2):
        features = {name: [0.0] for name in keys}
        features["cards"][0]=example[0]/7.0
        features["on_play"][0]=example[1]
        for card in example[2:]:
            features["opening_hand_"+card.replace(" ", "_").replace(",", "").replace("'", "")][0]+=1/7
        pred = sendPrediction(features)[0][0]
        mwr = mulliganWinRates[int(example[0]-1)][int(example[1])]
        result = getSuggestion(pred, mwr)
        result += "\nProbability of winning with this hand: "+str(pred*100)+"%"
        result += "\nProbability of winning after mulliganing: "+str(mwr*100)+"%"
        return result
    
    elif(len(example)>example[0]+2):#this could also do what the above does, but it's a little slower
        possibleHands = list(combinations(range(2, len(example)), int(example[0])))
        features = {name: [0.0 for i in range(len(possibleHands))] for name in keys}
        for i in range(len(possibleHands)):
            features["cards"][i]=example[0]/7
            features["on_play"][i]=example[1]
            indexes=list(possibleHands[i])
            for index in indexes:
                card=example[index]
                features["opening_hand_"+card.replace(" ", "_").replace(",", "").replace("'", "")][i]+=1/7
        preds = sendPrediction(features)
        #for i in range(len(possibleHands)):#remove this for loop to only show the best hand
        #    handIndices = list(possibleHands[i])
        #    hand = []
        #    for index in handIndices:
        #        hand.append(example[index])
        #    print("This hand: "+str(hand)+" has this probability of winning: "+str(preds[i]))
        bestIndex = argmax(preds)
        bestHandIndices = list(possibleHands[bestIndex])
        bestHand = []
        for index in bestHandIndices:
            bestHand.append(example[index])
        
        pred = preds[bestIndex][0]
        mwr = mulliganWinRates[int(example[0]-1)][int(example[1])]
        result = getSuggestion(pred, mwr)
        result += "\nBest combination of cards: "+(", ".join(bestHand))#This line technically returns some user input, it'd be tricky to inject through, but it should be escaped before sending to user
        result += "\nProbability of winning with this hand: "+str(pred*100)+"%"
        result += "\nProbability of winning after mulliganing: "+str(mwr*100)+"%"
        return result
    
    else:
        print("Something went seriously wrong here, the hand is missing some cards")

def argmax(preds):
    argmax = 0
    best = -1
    for i in range(len(preds)):
        if(preds[i]>best):
            best=preds[i]
            argmax=i
    return argmax

def getSuggestion(pred, mwr):
    if(pred>mwr*1.15):
        if(pred>.7):
            return "Snap keep"
        return "Keep"
    if(pred<mwr/1.15):
        return "Mulligan"
    if(pred>mwr*1.04):
        return "Keep, but it's close"
    if(pred<mwr/1.04):
        return "Mulligan, but it's close"
    if(pred>mwr):
        return "Keep, but it's very close"
    if(pred<mwr):
        return "Mulligan, but it's very close"
    return "Somehow, keeping and mulliganing are both exactly equally good, out to 12 decimal places of precision. Please contact Noah-R on GitHub, I would like to know how you possibly got this result."

def enterExample():
    hand = [float(input("Number of cards")), float(input("On play? 1 for yes, 0 for no."))]
    count = int(input("Number of cards to enter"))
    for i in range(count):
        hand.append(input("Enter card name"))
    return hand

def sendPrediction(features, url="http://localhost:8501/v1/models/model:predict"):
    data = json.dumps({"signature_name": "serving_default", "inputs": features})
    headers = {"content-type": "application/json"}
    response = requests.post(url, data=data, headers=headers)
    return json.loads(response.text)['outputs']

def setup(keys, cardnames, mwr):
    keys = eval(open(keys, "r").read())
    cardnames = open(cardnames, "r").read()
    mulliganWinRates = eval(open(mwr, "r").read())
    return keys, cardnames, mulliganWinRates