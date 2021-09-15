import pandas as pd
import json

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

def preprocess(filename):
    #read in data
    d = pd.read_csv(filename, header=0, usecols=range(0, 360))

    #read in sets
    sta = readSet("STA.json")
    stx = readSet("STX.json")

    #drop extraneous columns
    dropcols=["user_n_games_bucket", "draft_id", "build_index", "draft_time", "expansion", "event_type", "game_number", "opp_colors", "num_turns", "opp_num_mulligans", "rank", "opp_rank"]
    d = d.drop(dropcols, axis=1)

    d["lands"] = d.apply(lambda row: countCards(row, getCards([sta, stx], "type", "Land")), axis=1)

    return d

def marginal_preprocess():
    d = pd.read_csv("preprocessed_data.csv", header=0)
    columns = getAllCardColumns(d)
    d["Full Hand"] = d.apply(lambda row: rowToHand(row, columns), axis=1)
    return d

#data = preprocess("game_data_public.STX.PremierDraft.csv")
#data.to_csv("preprocessed_data.csv")
marginal_preprocess().to_csv("preprocessed_data.csv")