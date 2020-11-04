from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.algorithm import order_optimal_portfolio
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.factors import BusinessDaysSincePreviousEvent
from quantopian.pipeline.filters import Q500US
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.factors import BollingerBands
from quantopian.pipeline.factors import SimpleMovingAverage, EWMA
from quantopian.pipeline.factors import CustomFactor, RSI
from quantopian.pipeline.experimental import QTradableStocksUS
import talib as ta
import numpy as np
import math
import pandas as pd
import quantopian.optimize as opt


def initialize(context):
    #context.stop_price = 0
    context.stop_pct = 0.95
    
    # Any variables that we want to persist between function calls
    # should be stored in context instead of using global variables

    #context.market = sid(8554) #SPY
    #set_benchmark(context.market) #set own benchmark

    # Here define the commission and the slippage
    #Set commission: set the cost of our trade
    set_commission(commission.PerTrade(cost=0))
    #Slippage:calculates the realistic impact of our orders 
    #on the execution price we receive
    set_slippage(slippage.FixedSlippage(spread=0.00))
  

    #The schedule_function: lets your algorithm specify when methods are run by using date and/or time rules
    #Schedule balance function to run at the start of each day, when the market opens.
    schedule_function(
        my_rebalance,
        date_rules.every_day(),
        time_rules.market_open()
    )

    # Record variables at the end of each day.
    schedule_function(
        my_record_vars,
        date_rules.every_day(),
        time_rules.market_close()
    )

    # Create pipeline 
    pipe = make_pipeline()
    attach_pipeline(pipe, 'pipeline')
    
    # Maximum leverage：This will read leverage on every minute processed and chart the maximum every day
    context.mxlv = 0
    for i in range(1, 391):
        schedule_function(mxlv, date_rules.every_day(), time_rules.market_open(minutes=i))
     
 
      
def make_pipeline():
    '''
    Create pipeline.
    '''
    # Base universe 
    base_universe = QTradableStocksUS() 
    
    # Latest p/e ratio.
    pe_ratio = Fundamentals.pe_ratio.latest
    
    # Number of stocks invested long and short at the same time
    num = 60
    
    # Top 60 and bottom 60 stocks ranked by p/e ratio
    top_pe_stocks = pe_ratio.top(num, mask=base_universe)
    bottom_pe_stocks = pe_ratio.bottom(num, mask=base_universe)
    
    # Calculate the RSI indicator and the ranking for the stocks
    rsi = RSI(inputs=[USEquityPricing.close],
        window_length=14,
        mask=base_universe)
  
    oversold   = rsi < 30 and rsi.bottom(num)
    overbought = rsi > 70 and rsi.top(num)

    #mean = SimpleMovingAverage(inputs=[USEquityPricing.close],window_length=15, mask=base_universe)
    #mean_new = mean
    
    close_price =USEquityPricing.close.latest
         
    #calculate the Bollinger Bands:   
    upperBB, middleBB, lowerBB = BollingerBands(mask=base_universe, window_length=20, k=2)
    
    # Select securities to short
    shorts = overbought and upperBB < close_price and top_pe_stocks
    #shorts =upperBB < close_price and top_pe_stocks

    # Select securities to long
    longs = oversold and close_price < lowerBB and bottom_pe_stocks
    #longs =  close_price < lowerBB and bottom_pe_stocks

    # All securities that want to trade (fulfilling the criteria)
    securities_to_trade = (shorts | longs)

    return Pipeline(
        columns={
            'rsi': rsi,
            'upperBB' :upperBB,
            'lowerBB':lowerBB,
            'longs': longs,
            'shorts': shorts},
        screen=(securities_to_trade))

        
            
def before_trading_start(context, data):
    """
    Get pipeline results.
    """

    # Gets our pipeline output every day.
    pipe_results = pipeline_output('pipeline')

    # Go long in securities for which the 'longs' value is True,
    # and check if they can be traded. 
    context.longs = []
    for security in pipe_results[pipe_results['longs']].index.tolist():
        if data.can_trade(security):
            context.longs.append(security)

    # Go short in securities for which the 'shorts' value is True,
    # and check if they can be traded.
    context.shorts = []
    for security in pipe_results[pipe_results['shorts']].index.tolist():
        if data.can_trade(security):
            context.shorts.append(security)
            
    
def compute_target_weights(context, data):
    """
    Compute ordering weights.
    TargetWeights will prioritize zeroing positions that algorithm is trying to exit when it considers which target positions to adjust.
    """

    # Initialize empty target weights dictionary.
    # This will map securities to their target weight.
    weights = {}
    
    # If there are securities in our longs and shorts lists,
    # compute even target weights for each security. 
    if context.longs and context.shorts:
        long_weight = 0.5 / len(context.longs)
        short_weight = -0.5 / len(context.shorts)
    else:
        return weights
    
    # Exit positions in our portfolio if they are not in our longs or shorts lists. 
        
    #for security in context.portfolio.positions:
        #if security not in context.longs and security not in context.shorts and data.can_trade(security) and context.portfolio.returns <= 0.05:
            #weights[security] = 0
        #elif security not in context.longs and security not in context.shorts and data.can_trade(security) and context.portfolio.returns <= -0.05:
            #weights[security] = 0
        
    for security in context.portfolio.positions:
        set_trailing_stop(context, data)
        if security not in context.longs and security not in context.shorts and data.current(security, 'price') > context.stop_price:
            weights[security] = 0
        
    for security in context.longs:
        weights[security] = long_weight

    for security in context.shorts:
        weights[security] = short_weight 

    return weights  

def set_trailing_stop(context, data):
    for security in context.portfolio.positions:
        if context.portfolio.positions[security].amount:
            price = data.current(security, 'price')
            context.stop_price = context.stop_pct * price


def my_rebalance(context, data):
    """
    Rebalance daily.
    """
    
    # Calculate target weights to rebalance
    # So that everyday, half of our money will go long, and half of our money will go short.
    target_weights = compute_target_weights(context, data)
    
    # If we have target weights, rebalance our portfolio
    if target_weights:
        order_optimal_portfolio(
            objective=opt.TargetWeights(target_weights),#Create a TargetWeights object to set the order weights. This will become the ordering objective  
            constraints=[],
        )
        
        
def my_record_vars(context, data):
    '''
    Record variables at the end of each day.
    '''

    # Check how many long and short positions we have.
    longs = shorts = 0
    for position in context.portfolio.positions.values():
        if position.amount > 0:
            longs += 1
        elif position.amount < 0:
            shorts += 1

    # Record variables.
    record(
        leverage=context.account.leverage,        
        long_count=longs,                         
        short_count=shorts)      
    
def mxlv(context, data):
    if context.account.leverage > context.mxlv:
        context.mxlv = context.account.leverage
        record(MxLv = context.mxlv)