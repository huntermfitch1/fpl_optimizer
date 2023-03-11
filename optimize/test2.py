from fpl import FPL
import aiohttp
import asyncio
import pandas as pd
import numpy as np
from fixtures import get_gw_num_factor
from fixtures import get_fdr
from teams import *

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

fixture_dict = {id: asyncio.run(get_team_fixtures(id)) for id in range(1, 21)}
fixture_num_dict = get_gw_num_factor(fixture_dict[15], 4, 25)
print(fixture_num_dict)
