from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.filters import Q1500US 
import talib
import numpy
import math
import pandas

def initialize(context):
    context.stock = sid(698) #Boeing
    context.market = sid(8554) # market still SPY
    set_benchmark(context.stock) #set the stock as benchmark
    schedule_function(record_ADX, date_rules.every_day(), time_rules.market_close()) 
    
def record_ADX(context, data):   
    prices = data.history(context.stock, 'price', 252, '1d')
    market = data.history(context.market, 'price', 252, '1d')
    period = 14
    
    H = data.history(context.stock,'high', 2*period, '1d').dropna()
    L = data.history(context.stock,'low', 2*period, '1d').dropna()
    C = data.history(context.stock,'price', 2*period, '1d').dropna()
    
    ta_ADX = talib.ADX(H, L, C, period)
    ta_nDI = talib.MINUS_DI(H, L, C, period)
    ta_pDI = talib.PLUS_DI(H, L, C, period)
                 
    ADX = ta_ADX[-1]
    nDI = ta_nDI[-1]
    pDI = ta_pDI[-1]  
    
    beta = talib.BETA(prices, market, timeperiod=21)[-1:]
    
    if (ADX > 25) and (nDI > pDI):
            order_target_percent(context.stock, -1) and order_target_percent(context.market, beta)
    elif (ADX > 25) and (nDI < pDI):
            order_target_percent(context.stock, 1) and order_target_percent(context.market, -beta)
    
    record( ADX = ADX, nDI = nDI, pDI = pDI)