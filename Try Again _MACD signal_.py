#Author: Kory Hoang
#Developer: Mohammed Khalfan
#Email: hoang.kory@gmail.com

#Imports
from quantopian.pipeline.data.builtin import USEquityPricing
import statsmodels.api as sm 
import quantopian.pipeline.data 
import numpy as np
import pandas as pd
import talib
import scipy

def initialize(context):

    set_benchmark(symbol('GDXJ'))
    context.GDXJ = symbol('GDXJ')
    context.allocation = 1
    
    context.TakeProfitPct = 0.25
    context.StopLossPct = 0.05
    context.BuyPrice = 0
    
    context.bought = False
    context.sold = False

    # 30 min scheduler
    for x in [0,1,2,3,4,5]:
        schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=x, minutes=29))
        schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=x, minutes=59))
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_close())

    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())

    #Set commission and slippage
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

def my_rebalance(context,data):
    GDXJ_prices = data.history(context.GDXJ, "price", 10000, "1m").resample('30T',  closed='right', label='right') .last().dropna()
    #GDXJ_prices = data.history(context.GDXJ, "price", 100, "1d")
   
    ema12 = talib.EMA(GDXJ_prices,12)
    ema26 = talib.EMA(GDXJ_prices,26)
    macd = ema12 - ema26
    signal = talib.EMA(macd,9)
    
    record(SIG=macd[-1] - signal[-1])
    record(MACD=macd[-1])
    
    if macd[-2] < signal[-2] and macd[-1] >= signal[-1] and not context.bought:
        set_fixed_stop_long(context, data)
        order_target_percent(context.GDXJ, context.allocation)
        context.bought = True
        context.sold = False
        
    if macd[-1] < signal[-1] and not context.sold:
        set_fixed_stop_short(context, data)
        order_target_percent(context.GDXJ, -context.allocation)
        context.bought = False
        context.sold = True

def my_record_vars(context, data):    
    leverage = context.account.leverage
    #record(leverage=leverage)
    
def set_fixed_stop_long(context, data):
    #Only call this once when the stock is bought
    if data.can_trade(context.GDXJ):
        price = data.current(context.GDXJ, 'price')
        context.BuyPrice = price
        context.SellLossPrice= price - (context.StopLossPct * price)
        context.SellProfitPrice= (price * context.TakeProfitPct) + price
        
        
def set_fixed_stop_short(context, data):
    #Only call this once when the stock is bought
    if data.can_trade(context.GDXJ):
        price = data.current(context.GDXJ, 'price')
        context.BuyPrice = price
        context.SellLossPrice= price + (context.StopLossPct * price)
        context.SellProfitPrice= price - (price * context.TakeProfitPct)
    
def handle_data(context, data): 
    #If we have a position check sell conditions
    if context.portfolio.positions[context.GDXJ].amount != 0 and context.bought:
        price = data.current(context.GDXJ, 'price')
        
        if price > context.SellProfitPrice and len(get_open_orders()) == 0:
            order_target_percent(context.GDXJ, 0)
            context.bought = False
        if price < context.SellLossPrice and len(get_open_orders()) == 0:
            order_target_percent(context.GDXJ, 0)
            context.bought = False
            
    if context.portfolio.positions[context.GDXJ].amount != 0 and context.sold:
        price = data.current(context.GDXJ, 'price')
        
        if price < context.SellProfitPrice and len(get_open_orders()) == 0:
            order_target_percent(context.GDXJ, 0)
            context.sold = False
        if price > context.SellLossPrice and len(get_open_orders()) == 0:
            order_target_percent(context.GDXJ, 0)
            context.sold = False