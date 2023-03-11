from fpl import FPL
import aiohttp
import asyncio
import pandas as pd
import numpy as np
import pulp

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def get_team_fixtures(team_id):
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        team = await fpl.get_team(team_id)
        fixtures = await team.get_fixtures()
    return fixtures

