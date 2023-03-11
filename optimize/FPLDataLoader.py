from fpl import FPL
import aiohttp
import asyncio
import pandas as pd
import numpy as np
import pulp
from fixtures import get_fdr_adj
from fixtures import detect_gw_num

class FPLDataLoader:
    def __init__(self, apply_filters, current_gameweek) -> None:
        self.apply_filters = apply_filters
        self.current_gameweek = current_gameweek

    async def get_all_players(ret_json=True) -> dict:
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
        