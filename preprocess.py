import pandas as pd
import json

def getSet(setname):
    #Download the individual set file from MTGJSON, call passing setname as the set code as in the filename, returns a dictionary of card names mapped to card data dictionaries
    cardlist = json.loads(open(setname+".json", "rb").read())["data"]["cards"]
    setDict = {}
    for card in cardlist:
        setDict[card["name"]] = card
    return setDict

def rankToNumber(rank):
    #add these three lines to use rank
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
    #sta = getSet("STA")
    #stx = getSet("STX")
    
    #drop extraneous columns
    dropcols=["user_n_games_bucket", "draft_id", "build_index", "draft_time", "expansion", "event_type", "game_number", "opp_colors", "num_turns", "opp_num_mulligans", "rank", "opp_rank"]
    d = d.drop(dropcols, axis=1)

    return d

data = preprocess("game_data_public.STX.PremierDraft.csv")
data.to_csv("preprocessed_data.csv")