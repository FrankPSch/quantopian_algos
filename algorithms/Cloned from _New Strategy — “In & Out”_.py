# In and out strategy  

#Imports
import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor
import quantopian.optimize as opt  
import pandas as pd
import numpy as np

#Settings (variables and scheduling functions)
def initialize(context): 
    set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.00))
    context.daycounttot = 0 #count of total days since start
    context.in_out = 1 #start with being 'in' the market
    context.waitdays = 15 #min days of being out; 3 trading weeks
    context.outday = 0 #daycounttot when in_out=0
    context.spy_in = 0 #when SPY drops 30% go 'in' no matter what
    context.spyinday = 0 #daycounttot when spy_in=1
    context.SPY = symbol('SPY')
    context.XLI = symbol('XLI') #to capture drop in industrials
    context.DBB = symbol('DBB') #to capture drop in materials
    context.BIL = symbol('SHY') #to capture interest rate incr.
    context.alternative = [symbol('IEF'), symbol('TLT')] #investment if 'out' of the market
    algo.attach_pipeline(make_pipeline(context), 'pipeline')  
    # Schedule functions  
    schedule_function(  #daily check re getting 'out' of market
        check_out,  
        date_rules.every_day(),
        time_rules.market_open(minutes = 75)
    ) 
    schedule_function(  #weekly regular trading (getting 'in')
        trade,  
        date_rules.week_start(days_offset=4),
        time_rules.market_open(minutes = 75)
    ) 

class OwnMax(CustomFactor):
    inputs = [USEquityPricing.close]
    def compute(self, today, asset_ids, out, input_factor):
        # Calculates the column-wise max, ignoring NaNs
        out[:] = np.nanmax(input_factor, axis=0) 

class OwnShift(CustomFactor): 
    inputs = [USEquityPricing.close]
    def compute(self, today, assets, out, input_factor):  
        out[:] = input_factor[0] 
        
def make_pipeline(context): 
    out_xli = (USEquityPricing.close.latest / OwnShift(window_length=58) [context.XLI] - 1) < -0.07
    out_dbb = (USEquityPricing.close.latest / OwnShift(window_length=58) [context.DBB] - 1) < -0.07
    out_bil = (USEquityPricing.close.latest / OwnShift(window_length=58) [context.BIL] - 1) < -0.60/100
    in_spy = ((USEquityPricing.close.latest / OwnMax(window_length = 252) [context.SPY]) < 0.7) 
    
    pipe = Pipeline(columns={ 
            'out_xli': out_xli,
            'out_dbb': out_dbb,
            'out_bil': out_bil,
            'in_spy': in_spy,
            }, 
            )  
    return pipe

def check_out(context, data): #runs daily to determine w/h to exit
    df_big = algo.pipeline_output('pipeline')  
    
    #full weight on SPY, subject to the following conditions
    context.stock_weights = pd.Series(index=[context.SPY], data=1)
    
    #check whether XLI, DBB or BIL drop sign. (get out)
    if df_big.loc[context.XLI, 'out_xli'] | df_big.loc[context.DBB, 'out_dbb'] | df_big.loc[context.BIL, 'out_bil']:
        context.in_out = 0
        context.outday = context.daycounttot
    if context.daycounttot >= context.outday+context.waitdays:
        context.in_out = 1
        
    #check whether SPY drops sign. (stay in/go in)
    if df_big.loc[context.SPY, 'in_spy']:
        context.spy_in = 1
        context.spyinday = context.daycounttot
    if context.daycounttot >= context.spyinday+context.waitdays:
        context.spy_in = 0    
    
    #condition regarding getting out
    context.out = context.in_out==0 & context.spy_in==0
    if context.out:
        context.stock_weights = context.stock_weights*0
      
    #open alternative positions using remaining balance
    alt_weight = max(1.0 - context.stock_weights.sum(), 0) / len(context.alternative)  
    context.alt_weights = pd.Series(index=context.alternative, data=alt_weight)
    
    #buy alternative and sell SPY if getting out
    if context.out:
        total_weights = pd.concat([context.stock_weights, context.alt_weights])
        target_weights = opt.TargetWeights(total_weights)
        order_optimal_portfolio(  
            objective = target_weights,  
            constraints = []  
            )
    
    #increase total day counts by 1
    context.daycounttot = context.daycounttot+1

def trade(context, data): #runs weekly; trade/rebalance stocks
    total_weights = pd.concat([context.stock_weights, context.alt_weights])
    target_weights = opt.TargetWeights(total_weights)
    order_optimal_portfolio(
        objective = target_weights,  
        constraints = []  
        )