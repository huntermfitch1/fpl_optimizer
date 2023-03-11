from fpl import FPL
import aiohttp
import asyncio
import pandas as pd
import numpy as np
import pulp
from fixtures import get_fdr_adj
from fixtures import detect_gw_num

class PlayerLoad:
    def __init__(self, apply_filters, current_gameweek) -> None:
        self.apply_filter = apply_filters
        self.current_gameweek = current_gameweek
        self.all_players = asyncio.run(self.get_all_players())
        self.players = asyncio.run(self.get_players(self.all_players['id'].values.tolist()))
        self.player_df = pd.concat([self.all_players, self.compile_adj_scores(self.players)])

    async def get_all_players(self, ret_json=True) -> pd.DataFrame:
        session = aiohttp.ClientSession()
        fpl = FPL(session)
        all_players = await fpl.get_players(return_json = ret_json)
        await session.close()
        all_players = pd.json_normalize(all_players)
        return all_players

    async def get_players(self, pids) -> list:
        async with aiohttp.ClientSession() as session:
            fpl = FPL(session)
            players = await fpl.get_players(
                    pids,
                    include_summary=True)
        return players

    def get_gameweek_score(self, gw) -> int:
        return gw['total_points']

    def get_adjusted_score(self, player) -> int:
        recent_history = [gw for gw in player.history if gw['round'] > self.current_gameweek-10]
        last_10_gws = [self.get_gameweek_score(gw) for gw in recent_history]
        return sum([(1-(i/10))*(last_10_gws[i]) for i in range(len(last_10_gws))])

    def compile_adj_scores(self, players) -> pd.DataFrame:
        weighted_scores = [self.get_adjusted_score(player) for player in self.players]
        return pd.DataFrame(weighted_scores, columns=["weighted_score"])

    