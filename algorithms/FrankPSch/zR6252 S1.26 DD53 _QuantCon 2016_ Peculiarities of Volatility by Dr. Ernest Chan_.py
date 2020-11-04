import pandas as pd
import numpy as np
import math
from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor
from quantopian.pipeline.data.quandl import yahoo_index_vix
#from quantopian.pipeline.data.quandl import cboe_vix

NDays = 5

class ImpliedVolatility(CustomFactor):
    inputs = [yahoo_index_vix.close]
    #inputs = [cboe_vix]
    #window_length=NDays
    def compute(self, today, assets, out, vix):
        #out[:] = np.mean(vix, axis=0) # NB one identical column for every asset
        out[:] = np.mean(vix) # Does same as above, but more correct for ANTICIPATE CHANGE: https://www.quantopian.com/posts/upcoming-changes-to-quandl-datasets-in-pipeline-vix-vxv-etc-dot

        
def initialize(context):
    
    context.vxx = sid(38054)
    context.xiv = sid(40516)
    context.spy = sid(8554)
    # I just replaced XIV by QQQ, VXX by IEF in last Burrito Dan code and run back-test from 2007-06-01

    set_benchmark(context.xiv)
    
    pipe = Pipeline()
    attach_pipeline(pipe, 'example')
    
    iv = ImpliedVolatility(window_length=NDays)
    #iv = ImpliedVolatility()
    pipe.add(iv, 'iv')
    
    schedule_function(func=allocate, 
                      time_rule=time_rules.market_open(),
                      half_days=True)

    
def before_trading_start(context, data):

    output = pipeline_output('example')
    output = output.dropna() #makes a copy
    iv = output["iv"].iloc[0]
    hv = calculate_hv(context, data, NDays)  

    context.vrp = iv-hv

    record(ImpliedVolatility=iv)
    record(HistoricVolatility=hv)
    #record(VRP=iv-hv)
    

def calculate_hv(context, data, days):    
    close = data.history(context.spy, ["price"], days+1, "1d")
    close["ret"] = (np.log(close.price) - np.log(close.price).shift(1))
    return close.ret.std()*math.sqrt(252)*100

 
def allocate(context, data):
    if context.vrp > 0:
        order_target_percent(context.vxx, 0.00)
        order_target_percent(context.xiv, 1.00)
    else:
        order_target_percent(context.vxx, 1.00)
        order_target_percent(context.xiv, 0.00)