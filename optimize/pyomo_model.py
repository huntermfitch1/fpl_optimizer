from fpl import FPL
import aiohttp
import asyncio
import pandas as pd
import numpy as np
import pulp
from fixtures import get_fdr
from fixtures import get_gw_num_factor
from teams import *
import pyomo.environ as pyo

async def get_all_players(ret_json):
    session = aiohttp.ClientSession()
    fpl = FPL(session)
    players = await fpl.get_players(return_json = ret_json)
    await session.close()
    return players

async def get_players(pids):
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        players = await fpl.get_players(
                pids,
                include_summary=True)
    return players

def get_gameweek_score(player, gameweek, depth=10):
    points = 0
    for i, gw in enumerate(player.history[gameweek-depth:]):
        points += ((i+1)/depth) * gw["total_points"]
    return points

# base information about all players
players = asyncio.run(get_all_players(ret_json=True))
players = pd.json_normalize(players)

# weighted gameweek points
all_ids = players['id'].values.tolist()
ret = asyncio.run(get_players(all_ids))
weighted_scores = [get_gameweek_score(player, 27, depth = 5) for player in ret]

# cbind
players = pd.concat([players, pd.DataFrame(weighted_scores, columns=["weighted_score"])], axis=1)

# filter out some players and reindex
players = players[players.minutes > 500]
players = players[players.chance_of_playing_this_round != None]
players = players[players.chance_of_playing_this_round > 50.0]
players = players.reset_index(drop=True)
df = players[["first_name", 
                "web_name", 
                "team", 
                "now_cost", 
                "points_per_game", 
                "selected_by_percent",
                "element_type",
                "weighted_score"]]

expected_scores = df['weighted_score'].astype(float)
prices = df['now_cost'].astype(float) / 10.0
positions = df['element_type']
clubs = df['team']
names = df['web_name']
differentials = df['selected_by_percent'].astype(float)

# rename positions from int to str
position_dict = {1:'GK', 2:'DEF', 3:'MID', 4:'FWD'}
positions = [position_dict[i] for i in positions]

# weighted fixture difficulty
fixture_dict = {id: asyncio.run(get_team_fixtures(id)) for id in range(1, 21)}

fdr_dict = {id: get_fdr(fl, 5) for id, fl in fixture_dict.items()}
fixture_difficulty_weight = [fdr_dict[club] for club in clubs]

fixture_num_dict = {id: get_gw_num_factor(fl, 1, 27) for id, fl in fixture_dict.items()}
fixture_num_weight = [fixture_num_dict[club] for club in clubs]

expected_scores = expected_scores * fixture_num_weight

model = pyo.ConcreteModel()

# parameters
model.I = pyo.Set(initialize=list(range(len(expected_scores))))
model.C = pyo.Param(model.I, initialize=prices) # cost
model.S = pyo.Param(model.I, initialize=expected_scores) # score


# position parameters
def create_position_lists():
    pos_ind_dict = {}

    # creates one-hot encoded lists for each position
    for p in list(set(positions)):
        pos_ind_dict[p] = [1 if positions[i] == p else 0 for i in range(len(positions))]
    return pos_ind_dict

pos_dict = create_position_lists()

model.GK = pyo.Param(model.I, initialize=pos_dict['GK'])
model.DEF = pyo.Param(model.I, initialize=pos_dict['DEF'])
model.MID = pyo.Param(model.I, initialize=pos_dict['MID'])
model.FWD = pyo.Param(model.I, initialize=pos_dict['FWD'])

# team parameters
def create_team_lists():
    team_ind_dict = {}

    # creates one-hot encoded lists for each team
    for t in list(set(clubs)):
        team_ind_dict[t] = [1 if clubs[i] == t else 0 for i in range(len(clubs))]
    return team_ind_dict

team_dict = create_team_lists()

model.T1 = pyo.Param(model.I, initialize=team_dict[1])
model.T2 = pyo.Param(model.I, initialize=team_dict[2])
model.T3 = pyo.Param(model.I, initialize=team_dict[3])
model.T4 = pyo.Param(model.I, initialize=team_dict[4])
model.T5 = pyo.Param(model.I, initialize=team_dict[5])
model.T6 = pyo.Param(model.I, initialize=team_dict[6])
model.T7 = pyo.Param(model.I, initialize=team_dict[7])
model.T8 = pyo.Param(model.I, initialize=team_dict[8])
model.T9 = pyo.Param(model.I, initialize=team_dict[9])
model.T10 = pyo.Param(model.I, initialize=team_dict[10])
model.T11 = pyo.Param(model.I, initialize=team_dict[11])
model.T12 = pyo.Param(model.I, initialize=team_dict[12])
model.T13 = pyo.Param(model.I, initialize=team_dict[13])
model.T14 = pyo.Param(model.I, initialize=team_dict[14])
model.T15 = pyo.Param(model.I, initialize=team_dict[15])
model.T16 = pyo.Param(model.I, initialize=team_dict[16])
model.T17 = pyo.Param(model.I, initialize=team_dict[17])
model.T18 = pyo.Param(model.I, initialize=team_dict[18])
model.T19 = pyo.Param(model.I, initialize=team_dict[19])
model.T20 = pyo.Param(model.I, initialize=team_dict[20])

# variables
model.B = pyo.Var(model.I, domain=pyo.Binary)

# constraints

# player numbers
model.total_player_rule = pyo.Constraint(expr=sum(model.B[i] for i in model.I) == 15)

# total squad cost
model.cost_rule = pyo.Constraint(expr=sum(model.B[i]*model.C[i] for i in model.I) <= 100)



# position constraints
model.gk_rule = pyo.Constraint(expr=sum(model.B[i]*model.GK[i] for i in model.I) == 2)
model.def_rule = pyo.Constraint(expr=sum(model.B[i]*model.DEF[i] for i in model.I) == 5)
model.mid_rule = pyo.Constraint(expr=sum(model.B[i]*model.MID[i] for i in model.I) == 5)
model.fwd_rule = pyo.Constraint(expr=sum(model.B[i]*model.FWD[i] for i in model.I) == 3)

# club constraints
model.T1_rule = pyo.Constraint(expr=sum(model.B[i]*model.T1[i] for i in model.I) <= 3)
model.T2_rule = pyo.Constraint(expr=sum(model.B[i]*model.T2[i] for i in model.I) <= 3)
model.T3_rule = pyo.Constraint(expr=sum(model.B[i]*model.T3[i] for i in model.I) <= 3)
model.T4_rule = pyo.Constraint(expr=sum(model.B[i]*model.T4[i] for i in model.I) <= 3)
model.T5_rule = pyo.Constraint(expr=sum(model.B[i]*model.T5[i] for i in model.I) <= 3)
model.T6_rule = pyo.Constraint(expr=sum(model.B[i]*model.T6[i] for i in model.I) <= 3)
model.T7_rule = pyo.Constraint(expr=sum(model.B[i]*model.T7[i] for i in model.I) <= 3)
model.T8_rule = pyo.Constraint(expr=sum(model.B[i]*model.T8[i] for i in model.I) <= 3)
model.T9_rule = pyo.Constraint(expr=sum(model.B[i]*model.T9[i] for i in model.I) <= 3)
model.T10_rule = pyo.Constraint(expr=sum(model.B[i]*model.T10[i] for i in model.I) <= 3)
model.T11_rule = pyo.Constraint(expr=sum(model.B[i]*model.T11[i] for i in model.I) <= 3)
model.T12_rule = pyo.Constraint(expr=sum(model.B[i]*model.T12[i] for i in model.I) <= 3)
model.T13_rule = pyo.Constraint(expr=sum(model.B[i]*model.T13[i] for i in model.I) <= 3)
model.T14_rule = pyo.Constraint(expr=sum(model.B[i]*model.T14[i] for i in model.I) <= 3)
model.T15_rule = pyo.Constraint(expr=sum(model.B[i]*model.T15[i] for i in model.I) <= 3)
model.T16_rule = pyo.Constraint(expr=sum(model.B[i]*model.T16[i] for i in model.I) <= 3)
model.T17_rule = pyo.Constraint(expr=sum(model.B[i]*model.T17[i] for i in model.I) <= 3)
model.T18_rule = pyo.Constraint(expr=sum(model.B[i]*model.T18[i] for i in model.I) <= 3)
model.T19_rule = pyo.Constraint(expr=sum(model.B[i]*model.T19[i] for i in model.I) <= 3)
model.T20_rule = pyo.Constraint(expr=sum(model.B[i]*model.T20[i] for i in model.I) <= 3)

# objective
model.OBJ = pyo.Objective(expr=sum(model.B[i]*model.S[i] for i in model.I), sense=pyo.maximize)

solver = pyo.SolverFactory('glpk')
status = solver.solve(model)

for b in model.B:
    if pyo.value(model.B[b]) == 1:
        print(f"{names[b]} = {positions[b]}")

selections = [b for b in model.B if pyo.value(model.B[b]) == 1]

squad_positions = [positions[i] for i in range(len(positions)) if i in selections]
print(squad_positions)
squad_scores = [i for i in expected_scores if i in selections]
# positions
# expected_scores

def display_team(indexes):
    print("Starting 11 -")
    for i in indexes:
        max_gk_index = {'index': -1, 'score':0}
        if positions[i] == 'GK':
            if expected_scores[i] > max_gk_index['score']:
                max_gk_index['index'] = i
                max_gk_index['score'] = expected_scores[i]
        
        


# model.OBJ.pprint()
# model.B.pprint()

