import numpy as np
import pandas as pd
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

def trainTestSplit(d, split):
    if(split>1 or split<0 or type(d) != pd.DataFrame):
        return None
    splitpoint=int(len(d)*(1-split))
    return d[:splitpoint], d[splitpoint:]

def getMulliganWinRates(d, maxcards):#maxcards is the "cards" value for a 7 card hand, 1 after neural preprocessing, 7 after logistic preprocessing
    play = d[d["on_play"]==1.0]
    draw = d[d["on_play"]==0.0]
    print("Overall win rate on the play is "+str(play["won"].mean()))
    print("Overall win rate on the draw is "+str(draw["won"].mean()))
    play = play[play["cards"]<maxcards*6/6.8]
    draw = draw[draw["cards"]<maxcards*6/6.8]
    print("Win rate after mull to 6 on the play is "+str(play["won"].mean()))
    print("Win rate after mull to 6 on the draw is "+str(draw["won"].mean()))
    play = play[play["cards"]<maxcards*5/6.8]
    draw = draw[draw["cards"]<maxcards*5/6.8]
    print("Win rate after mull to 5 on the play is "+str(play["won"].mean()))
    print("Win rate after mull to 5 on the draw is "+str(draw["won"].mean()))
    play = play[play["cards"]<maxcards*4/6.8]
    draw = draw[draw["cards"]<maxcards*4/6.8]
    print("Win rate after mull to 4 on the play is "+str(play["won"].mean()))
    print("Win rate after mull to 4 on the draw is "+str(draw["won"].mean()))

def describe(path):
    pd.read_csv(path, header=0).describe().to_csv("describe.csv")

def logisticPreprocess(filename):
    #read in data
    d = pd.read_csv(filename, header=0, usecols=range(0, 359), index_col=False)

    #read in sets
    sta = readSet("STA.json")
    stx = readSet("STX.json")

    #drop extraneous columns
    dropcols=["user_win_rate_bucket", "user_n_games_bucket", "draft_id", "build_index", "draft_time", "expansion", "event_type", "game_number", "opp_colors", "num_turns", "opp_num_mulligans", "rank", "opp_rank"]
    d = d.drop(dropcols, axis=1)

    #crunch card type totals, this can probably be cleaned up into functions
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

    #one-hot encode land count
    d["two_lander"] = d["lands"].apply(lambda x:int(x == 2))
    d["three_lander"] = d["lands"].apply(lambda x:int(x == 3))
    d["four_lander"] = d["lands"].apply(lambda x:int(x == 4))
    d["five_lander"] = d["lands"].apply(lambda x:int(x == 5))
    d = d.drop("lands", axis=1)

    #convert num_mulligans to total cards
    d["cards"] = 7-d["num_mulligans"]
    d = d.drop("num_mulligans", axis=1)

    #convert boolean columns to int
    d["on_play"] = d["on_play"].apply(lambda x: int(x))
    d["won"] = d["won"].apply(lambda x: int(x))

    #drop all individual card columns, not going to use them for now, uncomment softenColumnNames when removing this
    d = d.drop(getAllCardColumns(d), axis=1)

    #remove _ ' , from column names, can only do this after all MTGJSON work is done
    #d = softenColumnNames(d)

    #shuffle
    d = d.reindex(np.random.permutation(d.index))

    return d

def neuralPreprocess(filename):
    #read in data
    d = pd.read_csv(filename, header=0, usecols=range(0, 359), index_col=False)

    #drop extraneous columns
    dropcols=["user_win_rate_bucket", "user_n_games_bucket", "draft_id", "build_index", "draft_time", "expansion", "event_type", "game_number", "opp_colors", "num_turns", "opp_num_mulligans", "rank", "opp_rank"]
    d = d.drop(dropcols, axis=1)

    #convert num_mulligans to total cards
    d["cards"] = 7-d["num_mulligans"]
    d = d.drop("num_mulligans", axis=1)

    #convert boolean columns to int, multiply by 7 so I can divide the whole dataframe by 7 later
    d["on_play"] = d["on_play"].apply(lambda x: int(x)*7)
    d["won"] = d["won"].apply(lambda x: int(x)*7)

    #divide the whole dataframe by 7, to normalize number of each card to other features
    d = d/7

    #remove _ ' , from column names
    d = softenColumnNames(d)

    #shuffle
    d = d.reindex(np.random.permutation(d.index))

    return d