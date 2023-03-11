from fpl import FPL
import aiohttp
import asyncio
import pandas as pd
import numpy as np
import pulp
from fixtures import get_fdr
from fixtures import get_gw_num_factor
from teams import *

from contextlib import contextmanager
import sys, os

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
    points_1_to_3 = 0
    points_4_to_6 = 0
    points_7_to_9 = 0
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

players2 = asyncio.run(get_all_players(ret_json=False))
# for p in players2:
#     print(get_gameweek_score(p, 21))

expected_scores = df['weighted_score'].astype(float)
prices = df['now_cost'].astype(float) / 10.0
positions = df['element_type']
clubs = df['team']
names = df['web_name']
differentials = df['selected_by_percent'].astype(float)

# weighted fixture difficulty
fixture_dict = {id: asyncio.run(get_team_fixtures(id)) for id in range(1, 21)}

fdr_dict = {id: get_fdr(fl, 5) for id, fl in fixture_dict.items()}
fixture_difficulty_weight = [fdr_dict[club] for club in clubs]

fixture_num_dict = {id: get_gw_num_factor(fl, 1, 27) for id, fl in fixture_dict.items()}
fixture_num_weight = [fixture_num_dict[club] for club in clubs]

expected_scores = expected_scores * fixture_num_weight
print(expected_scores)

def select_team(expected_scores, prices, positions, clubs, total_budget=100, sub_factor=0.2):
    num_players = len(expected_scores)
    model = pulp.LpProblem("Constrained value maximisation", pulp.LpMaximize)
    decisions = [
        pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    captain_decisions = [
        pulp.LpVariable("y{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    sub_decisions = [
        pulp.LpVariable("z{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]


    # objective function:
    model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * expected_scores[i]
                 for i in range(num_players)), "Objective"

    # cost constraint
    model += sum((decisions[i] + sub_decisions[i]) * prices[i] for i in range(num_players)) <= total_budget  # total cost

    # position constraints
    # 1 starting goalkeeper
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 1) == 1
    # 2 total goalkeepers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 1) == 2

    # 3-5 starting defenders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) <= 5
    # 5 total defenders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 2) == 5

    # 3-5 starting midfielders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) <= 5
    # 5 total midfielders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 3) == 5

    # 1-3 starting attackers
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) >= 1
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) <= 3
    # 3 total attackers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 4) == 3

    # club constraint
    for club_id in np.unique(clubs):
        model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if clubs[i] == club_id) <= 3  # max 3 players

    # differential constraint
    model += sum(decisions[i] for i in range(num_players) if differentials[i] < 10) == 2

    model += sum(decisions[i] for i in range(num_players) if differentials[i] < 5) == 1

    # model += sum(decisions[i] for i in range(num_players) if differentials[i] < 1) == 1 

    model += sum(decisions) == 11  # total team size
    model += sum(captain_decisions) == 1  # 1 captain
    
    for i in range(num_players):  
        model += (decisions[i] - captain_decisions[i]) >= 0  # captain must also be on team
        model += (decisions[i] + sub_decisions[i]) <= 1  # subs must not be on team

    model.solve()
    print("Total expected score = {}".format(model.objective.value()))

    return decisions, captain_decisions, sub_decisions

decisions, captain_decisions, sub_decisions = select_team(
        expected_scores.values,
        prices.values,
        positions.values,
        clubs.values
)
print("Starting 11")
for i in range(len(df)):
    if decisions[i].value() == 1:
        print(f"{names[i]:20s} {prices[i]} {positions[i]}")

print("")
print("Captain Decisions")
for i in range(len(df)):
    if captain_decisions[i].value() == 1:
        print(f"{names[i]:20s} {prices[i]} {positions[i]}")

print("")
print("Sub Decisions")
for i in range(len(df)):
    if sub_decisions[i].value() == 1:
        print(f"{names[i]:20s} {prices[i]} {positions[i]}")
