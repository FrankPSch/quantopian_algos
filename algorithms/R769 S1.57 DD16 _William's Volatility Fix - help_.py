#Author: Kory Hoang
#Developer: Mohammed Khalfan
#Email: hoang.kory@gmail.com

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

    context.wvf_length = 50
    context.ema1 = 5
    context.ema2 = 20
    context.StopLossPct = 0.03
    
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
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    set_long_only()
            
 
def market_open(context,data):
    if context.sell:
        context.sell = False
        if context.portfolio.positions[context.XIV].amount > 0 and len(get_open_orders()) == 0: 
            order_target_percent(context.XIV, 0)
            
def check_rsi(context,data):    
    
    vxx_highs = data.history(context.VXX, "high", context.wvf_length*2, "1d")
    vxx_prices = data.history(context.VXX, "price", context.wvf_length*2, "1d")
    vxx_lows = data.history(context.VXX, "low", context.wvf_length*2, "1d")
    vxx_highest = vxx_highs.rolling(window = context.wvf_length, center=False).max()
    WVF = ((vxx_highest - vxx_lows)/(vxx_highest)) * 100
    
    context.SmoothedWVF1 = talib.EMA(WVF, timeperiod=context.ema1) 
    context.SmoothedWVF2 = talib.EMA(WVF, timeperiod=context.ema2)
    SMA30VXX = talib.SMA(vxx_prices, 30)
    SMA5VXX = talib.SMA(vxx_prices, 5)
    
    record(WVF1=WVF[-1])
    #record(SMA30=SMA30VXX[-1]) USELESS WITHOUT VXX PRICE!
    record(EMA5=context.SmoothedWVF1[-1])
    record(EMA20=context.SmoothedWVF2[-1])
    print("VXX[-2]: {} VXX[-1]: {} SMA5: {} SMA30[-2]: {} SMA30[-1]: {} WVF[-2]: {} WVF[-1]: {} SmoothedWVF1[-2]: {} SmoothedWVF1[-1]: {} SmoothedWVF2[-2]: {} SmoothedWVF2[-1]: {}".format(vxx_prices[-2], vxx_prices[-1], SMA5VXX[-1], SMA30VXX[-2], SMA30VXX[-1], WVF[-2], WVF[-1], context.SmoothedWVF1[-2], context.SmoothedWVF1[-1], context.SmoothedWVF2[-2], context.SmoothedWVF2[-1]))
    
    ## BUY RULES
    if context.portfolio.positions[context.XIV].amount == 0:
        if ((WVF[-1] > context.SmoothedWVF1[-1] and WVF[-2] < context.SmoothedWVF1[-2] and WVF[-1] < context.SmoothedWVF2[-1]) or (WVF[-1] > context.SmoothedWVF1[-1] and WVF[-2] < context.SmoothedWVF1[-2] and WVF[-1] > context.SmoothedWVF2[-1] and WVF[-2] < context.SmoothedWVF2[-2])) and vxx_prices[-1] < SMA5VXX[-1]:
            set_fixed_stop(context, data)
            order_target_percent(context.XIV, 1.0)
            print("bought")
 
    ## SELL RULES
    if context.portfolio.positions[context.XIV].amount > 0 and len(get_open_orders()) == 0:
        #if vxx crosses above SMA30: sell
        if vxx_prices[-2] < SMA30VXX[-2] and vxx_prices[-1] > SMA30VXX[-1]:
            context.sell = True
            print("vxx crosses above SMA30: sell")
            
        #if wvf crosses under smoothwvf2: sell
        if WVF[-2] > context.SmoothedWVF2[-2] and WVF[-1] < context.SmoothedWVF2[-1]:
            context.sell = True
            print("wvf crosses under smoothwvf2: sell")

def set_fixed_stop(context, data):
    #Only call this once when the stock is bought
    if data.can_trade(context.XIV):
        price = data.current(context.XIV, 'price')
        context.BuyPrice = price
        context.SellLossPrice= price - (context.StopLossPct * price)
        
def handle_data(context, data): 
    #If we have a position check sell conditions
    if context.portfolio.positions[context.XIV].amount > 0:
        price = data.current(context.XIV, 'price')
        if price < context.SellLossPrice and len(get_open_orders()) == 0:
            order_target_percent(context.XIV, 0)
            context.sell = False
            print("SL hit")

def my_record_vars(context, data):    
    leverage = context.account.leverage
    record(leverage=leverage*10)