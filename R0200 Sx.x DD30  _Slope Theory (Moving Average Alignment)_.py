"""
##################################################################################################################
##################################################################################################################

John 1    King James Version (KJV)

1 In the beginning was the Word, and the Word was with God, and the Word was God.
2 The same was in the beginning with God.
3 All things were made by him; and without him was not any thing made that was made.
4 In him was life; and the life was the light of men.
5 And the light shineth in darkness; and the darkness comprehended it not.

##################################################################################################################
##################################################################################################################

Author: Leo Williams aka: D@rth V@d3r
Profession: Sr. Solution Architect and Software Engineer
Date: 05/08/2018
Email: lwilliams16@hotmail.com
phone: 678-348-6454 """


from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.quandl import cboe_vix as vix
from quantopian.pipeline import CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume, SimpleMovingAverage, RSI
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.data import morningstar
from brokers.ib import IBExchange
import talib
import math
import numpy as np
import statsmodels.api as smx

# GLOBAL STOCK FILTERS 

MIN_PRICE = 1                                   # MIN PRICE 
MAX_PRICE = 3                                   # MAX PRICE - (LOW PRICES HIGHER GAINS, BIGGER DD'S)
MAX_LEVERAGE = 1                                # MAX PORTFOLIO LEVERAGE
DOLLAR_VOLUME = 100000                          # LIQUITY FILTER
SHARES_OUTSTANDING = 20000000                   # TOTAL SHARES OUTSTANDING (FLOAT)
START_MINS = 15                                 # START TIME
TRAILING_STOP = .8                              # TRAILING STOP LOSS
MAX_LOSS = .9                                   # INITAL MAX STOP LOSS
VWAP_DISC = .98                                 # VWAP DISCOUNT * CURRENT PRICE

def initialize(context):
    # USE ROBIN HOOD OR GET KILLED IN TRADE FEES
    set_long_only()
    set_commission(commission.PerShare(cost=0.0, min_trade_cost=0.00))
    schedule_function(close_positions, date_rules.every_day(), time_rules.market_open())
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(minutes=START_MINS))
    attach_pipeline(make_pipeline(), 'my_pipeline')
    context.trail = {}
    context.pctx = 0
    context.vix = 30
    context.win = 0
    context.loss = 0
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    
    
def make_pipeline():
    
    pipe = Pipeline()
    pipe.add(GetVIX(inputs=[vix.vix_close]), 'VixClose')
    slp = Slope(window_length=100)
    dvol = AverageDollarVolume(window_length=30)
    so = morningstar.valuation.shares_outstanding.latest
    dv_screen = (dvol > DOLLAR_VOLUME)
    pclose = USEquityPricing.close.latest  
    #pe = morningstar.valuation_ratios.pe_ratio.latest
    #pb = morningstar.valuation_ratios.pb_ratio.latest
    
    #eps = (morningstar.valuation_ratios.earning_yield.latest * pclose)
    price_screenX = (pclose > MIN_PRICE)
    price_screenY = (pclose < MAX_PRICE)
    lq_screen = (so > SHARES_OUTSTANDING)
    slp_screen = (slp > 0)
    #eps_screen = (eps > 0)
    #pb_screen = (pb <= 1.5)
    #pe_screen = (pe < 15)
    
    
    pipe.set_screen(price_screenX & price_screenY & dv_screen & lq_screen & slp_screen)
    return pipe
 
def before_trading_start(context, data):
    context.output = pipeline_output('my_pipeline')
    context.security_list = context.output.index
    context.security_listF = {}
    try:
        context.vix = context.output["VixClose"].iloc[1]
    except:
       context.vix = 30
        
        
def my_rebalance(context,data):
    record(vix=context.vix)
    pc8 = data.history(context.security_list,'price',8,'1d')
    pc14 = data.history(context.security_list,'price',15,'1d')
    pc20 = data.history(context.security_list,'price',20,'1d')
    pc50 = data.history(context.security_list,'price',50,'1d')
    pc100 = data.history(context.security_list,'price',100,'1d')
    
    for s in context.security_list:
        cp = data.current(s,'price')
        rsi = talib.RSI(pc14[s],timeperiod=14)[-1]
        if slope(pc8[s]) > slope(pc20[s]) > slope(pc50[s]) > slope(pc100[s]) > 0:
            if rsi < 70 and context.vix < 20:
                context.security_listF[s] = s
                print s.symbol
    
    MAX_LEVERAGE = calc_leverage(context.portfolio.cash,2)
    context.pctx = diversify(context,data,context.security_listF,MAX_LEVERAGE)       
    if context.pctx == 0:
        return
    
    for s in context.security_listF:   
       if s not in context.portfolio.positions and data.can_trade(s) and s not in get_open_orders():
             cp = data.current(s,'price')
             if cp < (vwap(data,context,s) * VWAP_DISC):
                 tc(lp,[s,context.pctx, cp,False],[s.symbol + ' => Market Long','Error'])
                 context.trail[s] =  cp * MAX_LOSS
                 tc(so,[s,0,context.trail[s],False],['Stop Price set','Error'])
             else:
                 tc(lp,[s,context.pctx, vwap(data,context,s) * VWAP_DISC ,False],[s.symbol + ' => Market Long','Error'])
                 context.trail[s] =  vwap(data,context,s) * VWAP_DISC * MAX_LOSS 
                 tc(so,[s,0,context.trail[s],False],['Stop Price set','Error'])
                      
                 
def handle_data(context,data):
    pass

def close_positions(context,data):
    record(leverage=context.account.leverage)
    record(losses=context.loss)
    record(gains=context.win)
    for s in context.portfolio.positions:
      if s not in get_open_orders() and data.can_trade(s):
          
          cp = data.current(s,'price')
          cb = context.portfolio.positions[s].cost_basis
          pc100 = data.history(s,'price',100,'1d')
          mvg1 = talib.SMA(pc100,timeperiod=8)[-1]
          mvg2 = talib.SMA(pc100,timeperiod=20)[-2]
            
          if (cp * TRAILING_STOP) >= context.trail[s]:
            context.trail[s] = cp * TRAILING_STOP
            tc(so,[s,0,context.trail[s],False],['Stop Price raised','Error'])
          if mvg1 < mvg2:
            tc(mo,[s,0,False],['Stop Reached','Error'])  
            if cp > cb:
                context.win = context.win + 1
            if cp < cb:
                context.loss = context.loss + 1
            
            
def calc_leverage(cash,factor): 
    max_cash = 10000000
    if cash > max_cash:
        return max_cash/cash
    else:
        return cash/cash
        
        
def vwap(data,context,s):
    hist = data.history(s, fields=["price", "volume"], bar_count=390, frequency="1m")
    if hist["volume"].sum() > 0:
        print 
        return (hist["price"] * hist["volume"]).sum() / hist["volume"].sum()
    else:
        hist["price"][-1]
        
        
def diversify(context,data,watchlist,max_leverage):
        pcount = 0
        pctx = 0.00
        for s in watchlist:
            pcount += 1
        for s in watchlist:
            pcount += 1
        if pcount == 0:
            return 0
        pctx = float((1 - context.account.leverage)/float(pcount)) * max_leverage
        return pctx
        
def lo(stock,amount,price,*cancel):
        co(stock,cancel)
        order_target(stock,amount,style=LimitOrder(price,IBExchange.SMART))
def lp(stock,amount,price,*cancel):
        co(stock,cancel)
        order_target_percent(stock,amount,style=LimitOrder(price,IBExchange.SMART))
def sl(stock,amount,priceA,priceB,*cancel):
        co(stock,cancel)
        order_target(stock,amount,style=StopLimitOrder(priceA,priceB,IBExchange.SMART))
def so(stock,amount,price,*cancel):
        co(stock,cancel)
        order_target(stock,amount,style=StopOrder(price,IBExchange.SMART))     
def mo(stock,amount,*cancel):
        co(stock,cancel)
        order_target(stock,amount,style=MarketOrder(IBExchange.SMART))
def mp(stock,amount,*cancel):
        co(stock,cancel)
        order_target_percent(stock,amount,style=MarketOrder(IBExchange.SMART))
                          
def co(stock,*cancel):
    if cancel == True:
     for s in get_open_orders(stock):
        cancel_order(s)
        print s

def tc(fn,prms,msg):
    try:
        fn(*prms)
        print prms[0].symbol + ' => ' +  msg[0]
    except:
        print prms[0].symbol + ' => ' + msg[1]
    
class GetMaxHigh(CustomFactor):  
    window_length = 30
    inputs=[USEquityPricing.close] 
    def compute(self, today, assets, out, highs):
        out[:] = np.nanmax(highs, axis=0)

class Slope(CustomFactor):  
    inputs = [USEquityPricing.close]  
    def compute(self, today, assets, out, closes):  
        out[:] = slope(closes)
        
def slope(df):     
    return smx.OLS(df, smx.add_constant(range(-len(df) + 1, 1))).fit().params[-1]

class GetVIX(CustomFactor):  
    window_length = 1 
    def compute(self, today, assets, out, vix):  
        out[:] = np.nanmean(vix)