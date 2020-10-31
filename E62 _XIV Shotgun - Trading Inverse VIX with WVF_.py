#Author: Kory Hoang
#Developers: Jacob Lower & Mohammed Khalfan
#For collaboration requests, email: hoang.kory@gmail.com

#Imports
from quantopian.pipeline.data.builtin import USEquityPricing
import statsmodels.api as sm 
import quantopian.pipeline.data 
import numpy as np
import talib
import scipy
import math

def initialize(context):
    #set_benchmark(symbol('XIV'))
    context.XIV = symbol('XIV') 
    context.VXX = symbol('VXX')
    context.BOND = symbol('IEF')
    context.num_trades = 0

    context.rsi_length = 3
    context.rsi_trigger = 50
    context.wvf_length = 100
    context.ema1 = 10
    context.ema2 = 30
    context.TakeProfitPct = 0.25
    context.StopLossPct = 0.03
    context.XIVpct = 1.0
    context.BONDpct = 1.0
    
    context.BuyPrice = 0
    context.SellLossPrice = 0
    context.SellProfitPrice = 0
    context.sell = False

    #Schedules
    schedule_function(check_rsi, date_rules.every_day(), time_rules.market_close(), False)
    schedule_function(market_open, date_rules.every_day(), time_rules.market_open(), False)
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
    
    #Set commission and slippage
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0)) # FSC for IB

    set_long_only()
            
 
def market_open(context,data):
    if context.sell:
        context.sell = False
        if context.portfolio.positions[context.XIV].amount > 0 and len(get_open_orders()) == 0: 
            order_target_percent(context.XIV, 0)
            order_target_percent(context.BOND, context.BONDpct)
            
def check_rsi(context,data):    
    
    vxx_prices = data.history(context.VXX, "high", context.wvf_length*2, "1d")
    vxx_lows = data.history(context.VXX, "low", context.wvf_length*2, "1d")
    vxx_highest = vxx_prices.rolling(window = context.wvf_length, center=False).max()
    WVF = ((vxx_highest - vxx_lows)/(vxx_highest)) * 100

    rsi = talib.RSI(vxx_prices, timeperiod=context.rsi_length)
    
    context.SmoothedWVF1 = talib.EMA(WVF, timeperiod=context.ema1) 
    context.SmoothedWVF2 = talib.EMA(WVF, timeperiod=context.ema2)
    
    record(WVF1=WVF[-1])
    record(RSI=rsi[-1])
    record(EMA10=context.SmoothedWVF1[-1])
    record(EMA30=context.SmoothedWVF2[-1])
    
    print("vxx_low: ", vxx_lows[-1])
    print("vxx_highest: ", vxx_highest[-1])
    print("WVF: ", WVF[-1])
    print("context.SmoothedWVF1: ", context.SmoothedWVF1[-1])
    print("context.SmoothedWVF2: ", context.SmoothedWVF2[-1])
    
    ## BUY RULES
    #if WVF crosses over smoothwvf1 and wvf < smoothwvf2
    if ((WVF[-1] > context.SmoothedWVF1[-1] and WVF[-2] < context.SmoothedWVF1[-2] and WVF[-1] < context.SmoothedWVF2[-1]) or (context.SmoothedWVF1[-2] < context.SmoothedWVF2[-2] and context.SmoothedWVF1[-1] > context.SmoothedWVF2[-1]) or (WVF[-1] > context.SmoothedWVF1[-1] and WVF[-2] < context.SmoothedWVF1[-2] and WVF[-1] > context.SmoothedWVF2[-1] and WVF[-2] < context.SmoothedWVF2[-2])) and context.portfolio.positions[context.XIV].amount == 0:
        set_fixed_stop(context, data)
        order_target_percent(context.XIV, 1.0)
        order_target_percent(context.BOND, 0)
      
    ## SELL RULES
    if context.portfolio.positions[context.XIV].amount > 0 and len(get_open_orders()) == 0:
        #if rsi crosses over rsi_trigger
        if rsi[-2] < context.rsi_trigger and rsi[-1] > context.rsi_trigger:
            context.sell = True
            
        #if wvf crosses under smoothwvf2: sell
        elif WVF[-2] > context.SmoothedWVF2[-2] and WVF[-1] < context.SmoothedWVF2[-1]:
            context.sell = True

def set_fixed_stop(context, data):
    #Only call this once when the stock is bought
    if data.can_trade(context.XIV):
        price = data.current(context.XIV, 'price')
        context.BuyPrice = price
        context.SellLossPrice= price - (context.StopLossPct * price)
        context.SellProfitPrice= (price * context.TakeProfitPct) + price
        
def handle_data(context, data): 
    #If we have a position check sell conditions
    if context.portfolio.positions[context.XIV].amount > 0:
        price = data.current(context.XIV, 'price')
        if price > context.SellProfitPrice and len(get_open_orders()) == 0:
            order_target_percent(context.XIV, 0)
            order_target_percent(context.BOND, context.BONDpct)
            context.sell = False
        if price < context.SellLossPrice and len(get_open_orders()) == 0:
            order_target_percent(context.XIV, 0)
            order_target_percent(context.BOND, context.BONDpct)
            context.sell = False

def my_record_vars(context, data):    
    leverage = context.account.leverage
    record(leverage=leverage)