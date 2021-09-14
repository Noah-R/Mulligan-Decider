import pandas as pd

def rankToNumber(rank):
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
    d = pd.read_csv(filename, header=0)
    dropcols=["user_n_games_bucket", "draft_id", "build_index", "draft_time", "expansion", "event_type", "game_number", "opp_colors", "num_turns", "opp_num_mulligans"]
    for col in d.keys():
        if("drawn_" in col):
            dropcols.append(col)
    d = d.drop(dropcols, axis=1)
    d["rank"] = d["rank"].apply(lambda x: rankToNumber(x))
    d["opp_rank"] = d["opp_rank"].apply(lambda x: rankToNumber(x))
    d["rank_differential"] = d.apply(lambda row: d["rank"] - d["opp_rank"], axis=1)

    return d

data = preprocess("game_data_public.STX.PremierDraft.csv")
data.to_csv("preprocessed_data.csv")