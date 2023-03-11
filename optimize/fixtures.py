from fpl import FPL
import aiohttp
import asyncio
import pandas as pd
import numpy as np
import pulp

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def get_gw_num_factor(fixture_list, num_gws, cur_gw):
    gws = list(range(cur_gw, cur_gw + num_gws))  # list of gameweeks to consider
    num_gws_dict = {gws[i]: (sum([fixture_list[j]['event'] == gws[i] for j in range(num_gws)])) for i in range(len(gws))}
    weights = [round(1-((1/num_gws)*i), 2) for i in range(num_gws)]
    nums = [num_gws_dict[gw] for gw in gws]
    res_list = [weights[i] * nums[i] for i in range(len(weights))]
    return sum(res_list)

def get_fdr(fixture_list, num_gws) -> dict:
    tot_score = sum(fixture_list[i]['difficulty'] for i in range(num_gws))
    return round(3 * num_gws / tot_score, 2)