import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import json

def readSet(setname):
    #First download the individual set file from MTGJSON, then call this function passing filename, returns a dictionary of card names mapped to card data dictionaries
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

def rowToHand(row, columns):#deprecated, takes way too long to run as written
    cards=[]
    if(row.index%10000==0):
        print(row.index)
    for colName in columns:
        for num in range(int(row[colName])):
            cards.append(colName.replace("opening_hand_", ""))
    return cards

def getColumns(sets, condition):
    if type(sets) is not list:
        sets=[sets]
    cols=[]
    for cardSet in sets:
        for card in cardSet:
            if(eval(condition)):
                cardname="opening_hand_"+card
                if("//" in cardname):
                    cardname=cardname[:cardname.index("//")-1]
                cols.append(cardname)
    return cols

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
    d = pd.read_csv(filename, header=0, usecols=range(0, 359), index_col=False)

    #read in sets
    sta = readSet("STA.json")
    stx = readSet("STX.json")

    #drop extraneous columns
    dropcols=["user_win_rate_bucket", "user_n_games_bucket", "draft_id", "build_index", "draft_time", "expansion", "event_type", "game_number", "opp_colors", "num_turns", "opp_num_mulligans", "rank", "opp_rank"]
    d = d.drop(dropcols, axis=1)
    
    #convert boolean columns to int
    d["on_play"] = d["on_play"].apply(lambda x: int(x))
    d["won"] = d["won"].apply(lambda x: int(x))

    #crunch card type totals, this can be cleaned up into functions once it works
    cards=getColumns([sta, stx], "\"Land\" in cardSet[card][\"type\"]")
    d["lands"] = d.loc[:, cards].sum(axis=1)
    cards=getColumns([sta, stx], "cardSet[card][\"convertedManaCost\"] == 1")
    d["ones"] = d.loc[:, cards].sum(axis=1)
    cards=getColumns([sta, stx], "cardSet[card][\"convertedManaCost\"] == 2")
    d["twos"] = d.loc[:, cards].sum(axis=1)
    cards=getColumns([sta, stx], "cardSet[card][\"convertedManaCost\"] == 3")
    d["threes"] = d.loc[:, cards].sum(axis=1)
    cards=getColumns([sta, stx], "cardSet[card][\"convertedManaCost\"] == 4")
    d["fours"] = d.loc[:, cards].sum(axis=1)
    cards=getColumns([sta, stx], "cardSet[card][\"convertedManaCost\"] == 5")
    d["fives"] = d.loc[:, cards].sum(axis=1)
    cards=getColumns([sta, stx], "cardSet[card][\"convertedManaCost\"] == 6")
    d["sixes"] = d.loc[:, cards].sum(axis=1)
    cards=getColumns([sta, stx], "cardSet[card][\"convertedManaCost\"] == 7")
    d["sevens"] = d.loc[:, cards].sum(axis=1)
    cards=getColumns([sta, stx], "cardSet[card][\"convertedManaCost\"] == 8")
    d["eights"] = d.loc[:, cards].sum(axis=1)

    #drop all individual card columns, not going to use them for now, uncomment softenColumnNames when removing this
    d = d.drop(getAllCardColumns(d), axis=1)

    #remove _ ' , from column names, can only do this after all MTGJSON work is done
    #d = softenColumnNames(d)

    #shuffle
    d = d.reindex(np.random.permutation(d.index))

    return d

def describe(path):
    pd.read_csv(path, header=0).describe().to_csv("describe.csv")


data = preprocess("game_data_public.STX.PremierDraft.csv")
data.to_csv("preprocessed_data.csv")

target="won"
learningrate=.01
batchsize=10
epochs=25

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
    x = features,
    y = label,
    batch_size = batchsize,
    epochs = epochs,
    shuffle = True,
    verbose = 2,
    validation_split = 0.1
)

model.save("model_9_19_2021")


#WARNING:tensorflow:Layers in a Sequential model should only have a single input tensor, but we receive a <class 'dict'>... Consider rewriting this model with the Functional API.