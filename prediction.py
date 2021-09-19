import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import json
from matplotlib import pyplot as plt

def readSet(setname):
    #Download the individual set file from MTGJSON, call passing filename, returns a dictionary of card names mapped to card data dictionaries
    cardlist = json.loads(open(setname, "rb").read())["data"]["cards"]
    setDict = {}
    for card in cardlist:
        setDict[card["name"]] = card
    return setDict

def getAllCardColumns(dataset):
    columns=[]
    for name in dataset.keys():
        if("opening_hand_" in name):
            columns.append(name)
    return columns

def rowToHand(row, columns):
    cards=[]
    if(row.index%10000==0):
        print(row.index)
    for colName in columns:
        for num in range(int(row[colName])):
            cards.append(colName.replace("opening_hand_", ""))
    return cards

def countCards(row, cardList):
    count=0
    for card in cardList:
        if("opening_hand_"+card in row.keys()):
            count+=int(row["opening_hand_"+card])
    return count

def getCards(sets, field, value):#make this work for conjunctions of multiple fields and values
    if type(sets) is not list:
        sets=[sets]
    hits={}
    for cardSet in sets:
        for card in cardSet:
            if(value in cardSet[card][field]):
                hits[card]=cardSet[card]
    return hits

def rankToNumber(rank):
    #to add rank to model, add these lines to preprocessing, and remove relevant columns from dropcols
    #d["rank"] = d["rank"].apply(lambda x: rankToNumber(x))
    #d["opp_rank"] = d["opp_rank"].apply(lambda x: rankToNumber(x))
    #d["rank_differential"] = d.apply(lambda row: d["rank"] - d["opp_rank"], axis=1)

    if(type(rank) is not str):
        return 0
    if("Mythic" in rank):
        return 6
    num=(4-int(rank[rank.index("-")+1]))*.25
    if("Diamond" in rank):
        num+=5
    if("Platinum" in rank):
        num+=4
    if("Gold" in rank):
        num+=3
    if("Silver" in rank):
        num+=2
    if("Bronze" in rank):
        num+=1
    return num

def softenColumnNames(data):
    columns=[]
    for name in data.keys():
        columns.append(name.replace(" ", "_").replace(",", "").replace("'", ""))
    data.columns=columns
    return data

def preprocess(filename):
    #read in data
    d = pd.read_csv(filename, header=0, usecols=range(0, 360))

    #read in sets
    sta = readSet("STA.json")
    stx = readSet("STX.json")

    #drop extraneous columns
    dropcols=["user_n_games_bucket", "draft_id", "build_index", "draft_time", "expansion", "event_type", "game_number", "opp_colors", "num_turns", "opp_num_mulligans", "rank", "opp_rank"]
    d = d.drop(dropcols, axis=1)
    
    #convert boolean columns to int
    d["on_play"] = d["on_play"].apply(lambda x: int(x))
    d["won"] = d["won"].apply(lambda x: int(x))

    #crunch land total
    d["lands"] = d.apply(lambda row: countCards(row, getCards([sta, stx], "type", "Land")), axis=1)

    #remove _ ' , from column names, can only do this after all MTGJSON work is done
    #d = softenColumnNames(d)

    return d

def describe(path):
    pd.read_csv(path, header=0).describe().to_csv("describe.csv")


#data = preprocess("game_data_public.STX.PremierDraft.csv")
#data.to_csv("preprocessed_data.csv")

data = pd.read_csv("preprocessed_data.csv", usecols=[0, 1, 2, 3, 4, 349], header=0).drop("Unnamed: 0", axis=1)#delete usecols to include individual card counts, but it'll take you forever

#temporarily done on-the-fly, save this to the preprocessed data file in the future
data = softenColumnNames(data)
#end

target="won"
learningrate=.001
batchsize=10
epochs=50

features=[]

for col in data.keys():
    if(col!=target):
        features.append(tf.feature_column.numeric_column(col))

model = tf.keras.models.Sequential([
    layers.DenseFeatures(features),
    layers.Dense(units=1, input_shape=(1,) , activation=tf.sigmoid)
])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=learningrate),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["mae"]
)

features = {name: np.array(value) for name, value in data.items()}
label = np.array(features.pop(target)) 

model.fit(
    x=features,
    y=label,
    batch_size=batchsize,
    epochs=epochs,
    shuffle=True,
    verbose=2
)

"""
WARNING:tensorflow:Layers in a Sequential model should only have a single input tensor, but we receive a <class 'dict'>... Consider rewriting this model with the Functional API.
"""