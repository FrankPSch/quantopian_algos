from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor
from quantopian.pipeline.data.accern import alphaone_free as alphaone

import pandas as pd
import numpy as np


# Calculates the average impact of the sentiment over the window length
class AvgImpact(CustomFactor):
    
    def compute(self, today, assets, out, sentiment, impact):
        np.mean((sentiment*impact), axis=0, out=out)

        
class AvgDailyDollarVolumeTraded(CustomFactor):
    
    inputs = [USEquityPricing.close, USEquityPricing.volume]
    window_length = 20
    
    def compute(self, today, assets, out, close_price, volume):
        out[:] = np.mean(close_price * volume, axis=0)

        
# Put any initialization logic here.  The context object will be passed to
# the other methods in your algorithm.
def initialize(context):
    
    pipe = Pipeline()
    pipe = attach_pipeline(pipe, name='sentiment_metrics')
    
    # Add our AvgImpact factor to the pipeline
    pipe.add(AvgImpact(inputs=[alphaone.article_sentiment, alphaone.impact_score], window_length=7), "avg_impact")    
    
    dollar_volume = AvgDailyDollarVolumeTraded()
    
    # Screen out low liquidity securities.
    pipe.set_screen(dollar_volume > 10**7)
    context.shorts = None
    context.longs = None
    # context.spy = sid(8554)
    
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open(hours=1))
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))
    set_slippage(slippage.FixedSlippage(spread=0))
    
    
def before_trading_start(context, data):
    results = pipeline_output('sentiment_metrics').dropna()
    longs = results[results['avg_impact'] > 0]
    shorts = results[results['avg_impact'] < 0]
    long_ranks = longs["avg_impact"].rank().order()
    short_ranks = shorts['avg_impact'].rank().order()
    short_quartile = int(len(short_ranks.index)*.01)
    long_quartile = int(len(long_ranks.index)*.01)
    context.shorts = short_ranks.tail(short_quartile)
    context.longs = long_ranks.head(long_quartile)
    
    # The pipe character "|" is the pandas union operator
    update_universe(context.longs.index | context.shorts.index)
    

# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):
    record(lever=context.account.leverage,
           exposure=context.account.net_leverage,
           num_pos=len(context.portfolio.positions),
           oo=len(get_open_orders()))
        
    
def rebalance(context, data):
    
    for security in context.shorts.index:
        if get_open_orders(security):
            continue
        if security in data:
            order_target_percent(security, -1.0 / len(context.shorts))
            
    for security in context.longs.index:
        if get_open_orders(security):
            continue
        if security in data:
            order_target_percent(security, 1.0 / len(context.longs))
            
    for security in context.portfolio.positions:
        if get_open_orders(security):
            continue
        if security in data:
            if security not in (context.longs.index | context.shorts.index):
                order_target_percent(security, 0)