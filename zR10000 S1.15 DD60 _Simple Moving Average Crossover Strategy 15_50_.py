"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.filters import Q1500US 

# -----------------------------------------------------------------------  
stock, bond, ma_f, ma_s, lev = symbol('AAPL'), symbol('TLT'), 55, 56, 1.0  
# -----------------------------------------------------------------------  

def initialize(context):  
    set_benchmark(stock)  
    schedule_function(trade,date_rules.every_day(),time_rules.market_open(minutes = 1)) 
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB


def trade(context,data):  
    if get_open_orders(): return  
    mavg_f = data.history(stock,'price',ma_f,'1d').mean()  
    mavg_s = data.history(stock,'price',ma_s,'1d').mean()

    if all(data.can_trade([stock, bond])):  
        if mavg_f > mavg_s:  
            order_target_percent(stock, lev*1.9)  
            order_target_percent(bond,  0.0)  
        elif mavg_f < mavg_s:  
            order_target_percent(stock, 0.7)  
            order_target_percent(bond,  lev*0.3)

         

def before_trading_start(context, data):  
    record(Leverage = context.account.leverage)