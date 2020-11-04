from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor
from quantopian.pipeline.data.quandl import cboe_vix

import numpy as np
import pandas as pd


def rename_col(df):
    df = df.rename(columns={'Close': 'price','Trade Date': 'Date'})
    df = df.fillna(method='ffill')
    df = df[['price', 'Settle','sid']]
    # Shifting data by one day to avoid forward-looking bias
    return df.shift(1)

 
def initialize(context):
    
    set_commission(commission.PerShare(cost = 0.0035, min_trade_cost = .35))
    #set_slippage(slippage.FixedSlippage(spread=0)) 
    context.leverage = 1.96
    """
    Called once at the start of the algorithm.
    """
    #securities we are investing in
    
    context.XIV = symbol('XIV')
    context.VXX = symbol('VXX')
    context.SHY = symbol('LQD')
    context.SPY = symbol('SPY')
    
    my_pipe = Pipeline()
    attach_pipeline(my_pipe, 'my_pipeline')
    my_pipe.add(GetVIX(inputs=[cboe_vix.vix_open]), 'VixOpen')
    
    
    # Front month VIX futures data
    fetch_csv('https://www.quandl.com/api/v1/datasets/CHRIS/CBOE_VX1.csv?api_key=kBrKnGzVAWhbLJvGu-NB', 
        date_column='Trade Date', 
        date_format='%Y-%m-%d',
        symbol='v1',
        post_func=rename_col)
    # Second month VIX futures data
    fetch_csv('https://www.quandl.com/api/v1/datasets/CHRIS/CBOE_VX2.csv?api_key=kBrKnGzVAWhbLJvGu-NB', 
        date_column='Trade Date', 
        date_format='%Y-%m-%d',
        symbol='v2',
        post_func=rename_col)
   

    
    # Rebalance every day, 1 minute after market open.
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(minutes=2))
   
    schedule_function(record_leverage, date_rules.every_day())
   
def record_leverage(context, data):
    record(leverage = context.account.leverage)
 
def before_trading_start(context, data):
    """
    Called every day before market open.
    """ 
    context.output = pipeline_output('my_pipeline')     
    context.vix = context.output["VixOpen"].iloc[0]
    
class GetVIX(CustomFactor):
    window_length = 1
    def compute(self, today, assets, out, vix):
        out[:] = vix[-1]
     
 
def my_rebalance(context,data):
    """
    Execute orders according to our schedule_function() timing. 
    """
    last_ratio = data.current('v1','Settle')/data.current('v2','Settle')
    threshold_bot = 0.9
    threshold_top = 0.95
    ex_top = 1.1
    ex_bot = .856
    vr_top = 19.95
    vr_bot = 12
    vix_top = 24
    vix_bot = 12.25
    
    if data.can_trade(context.XIV) and data.can_trade(context.SPY):
        
        #VIX Opens above 24 and extreme backwardization = VXX w/ Leverage
        
        if context.vix > vix_top and last_ratio > ex_top:
            order_target_percent(context.XIV, 0)
            order_target_percent(context.VXX, 1 * context.leverage)#lev
            order_target_percent(context.SHY, 0)
            order_target_percent(context.SPY, 0)
            
        #VIX Opens above 24 and backwardization = Cash    
        
        elif context.vix >= vix_top and last_ratio >= threshold_top:
            order_target_percent(context.XIV, 0)
            order_target_percent(context.VXX, 0)
            order_target_percent(context.SHY, 0)
            order_target_percent(context.SPY, 0)
            
        elif context.vix <= vix_bot and context.vix >= vr_bot and last_ratio > threshold_bot:
            order_target_percent(context.XIV, 0)
            order_target_percent(context.VXX, 0)
            order_target_percent(context.SHY, 0)
            order_target_percent(context.SPY, 0)
            
        elif context.vix > vr_top and last_ratio > threshold_bot and last_ratio < threshold_top:
            order_target_percent(context.XIV, .5 * context.leverage)#lev
            order_target_percent(context.VXX, .5)#
            order_target_percent(context.SHY, 0)
            order_target_percent(context.SPY, 0)
            
        #vix is very low and we are not in contango, we suppose vix will raise
        elif context.vix <= vix_bot and last_ratio < ex_bot:
            order_target_percent(context.XIV, 0)
            order_target_percent(context.VXX, 1 * context.leverage)#
            order_target_percent(context.SHY, 0)
            order_target_percent(context.SPY, 0)
            
        #vix is high and we are not in a situation in which the things are turning bad
        elif context.vix >= vix_top and last_ratio < threshold_top:
            order_target_percent(context.XIV, 1 * context.leverage)#lev
            order_target_percent(context.VXX, 0)
            order_target_percent(context.SHY, 0)
            order_target_percent(context.SPY, 0)
                    
        #vix is in his standard range and we are in contango 
        elif context.vix > vix_bot and context.vix < vix_top and last_ratio < threshold_bot:
            order_target_percent(context.XIV, 1 * context.leverage)#lev
            order_target_percent(context.VXX, 0)
            order_target_percent(context.SHY, 0)
            order_target_percent(context.SPY, 0)
           
        #vix is in his standard range but we are not in contango, we wait switching to bonds
        elif context.vix > vix_bot and context.vix < vix_top and last_ratio > threshold_bot and last_ratio < threshold_top:
            order_target_percent(context.XIV, 0)
            order_target_percent(context.VXX, 0)
            order_target_percent(context.SHY, 1 * context.leverage)#?lev
            order_target_percent(context.SPY, 0)
            
       #when there is conflict between signals we switch to bonds
        elif context.vix < vix_bot and last_ratio < threshold_bot:
            order_target_percent(context.XIV, 0)
            order_target_percent(context.VXX, 0)
            order_target_percent(context.SHY, 1)#
            order_target_percent(context.SPY, 0)
            
    record(XIV=context.portfolio.positions[symbol('XIV')].amount)
    record(LQD=context.portfolio.positions[symbol('LQD')].amount)
    record(VXX=context.portfolio.positions[symbol('VXX')].amount)
    
    #record(Cash=context.portfolio.cash)
    #record(VIX=context.vix)
    #record(ratio=last_ratio)