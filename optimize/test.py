from fpl import FPL
import aiohttp
import asyncio
import pandas as pd
import numpy as np
import pulp

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def get_players(pids):
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        players = await fpl.get_players(
                pids,
                include_summary=True)
    return players

def get_gameweek_score(player, gameweek):
    points_1_to_3 = 0
    points_4_to_6 = 0
    points_7_to_9 = 0
    for gw in player.history:
        for i in range(1, 4):
            if (gw["round"] == gameweek-i):
                points_1_to_3 += gw["total_points"]
        for i in range(4, 7):
            if (gw['round'] == gameweek-i):
                points_4_to_6 += gw["total_points"]
        for i in range(7, 10):
            if (gw['round'] == gameweek-i):
                points_7_to_9 += gw["total_points"]
    return round((points_1_to_3) + (0.5*points_4_to_6) + (0.2*points_7_to_9), 0)
                
    return tot_points

async def get_all_players(ret_json):
    session = aiohttp.ClientSession()
    fpl = FPL(session)
    players = await fpl.get_players(return_json = ret_json)
    await session.close()
    return players

players = asyncio.run(get_all_players(ret_json=True))
players = pd.json_normalize(players)

all_ids = players['id'].values.tolist()
ret = asyncio.run(get_players(all_ids))
weighted_scores = [get_gameweek_score(player, 25) for player in ret]
print(weighted_scores)

