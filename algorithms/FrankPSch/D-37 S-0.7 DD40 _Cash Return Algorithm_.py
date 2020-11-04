from __future__ import division
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.experimental import QTradableStocksUS
from quantopian.pipeline.factors import CustomFactor
import quantopian.optimize as opt
import quantopian.algorithm as algo

from scipy import stats
import numpy as np
import pandas as pd
import math

from quantopian.pipeline.data import Fundamentals
        
def make_pipeline():
    cash_return = Fundamentals.cash_return.latest
    universe = QTradableStocksUS() & cash_return.notnull()
    
    ranking = cash_return.rank(mask=universe)
    
    longs = ranking.percentile_between(95, 100)
    shorts = ranking.percentile_between(1, 5)
    
    long_short_screen = (longs | shorts)
    
    pipe = Pipeline(columns = {
        'longs':longs,
        'shorts':shorts,     
    },
    screen = long_short_screen)
    return pipe

def initialize(context):
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    
    attach_pipeline(make_pipeline(), 'my_pipe')
    
    schedule_function(rebalance,
                      date_rules.every_day(),
                      time_rules.market_open(minutes=7))
    
    schedule_function(record_var,
                      date_rules.every_day(),
                      time_rules.market_close(minutes=7))
    
def before_trading_start(context, data):
    context.output = pipeline_output('my_pipe')
    context.longs = context.output[context.output.longs]
    context.shorts = context.output[context.output.shorts]
    context.security_list = context.shorts.index.union(context.longs.index)
    
def rebalance(context, data):
    weight = 1.00000 / len(context.security_list)
    
    for stock in context.security_list:
        if stock in context.longs.index:
            order_target_percent(stock,  weight)
            
        elif stock in context.shorts.index:
            order_target_percent(stock, -weight)
    
    for stock in context.portfolio.positions:
        if stock not in context.security_list:
            order_target_percent(stock, 0)            
    
def record_var(context, data):
    record(leverage = context.account.leverage)