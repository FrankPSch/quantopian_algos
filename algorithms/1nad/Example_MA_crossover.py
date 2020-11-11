''' 
TIPS:
For recorded values see: Full Backtest/Activity/Custom Data
'''
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline, CustomFactor
import quantopian.pipeline.filters as Filters
import quantopian.pipeline.factors as Factors
import pandas as pd
import numpy as np
from quantopian.pipeline.data.builtin import USEquityPricing


MY_STOCKS = symbols('AAPL')
WEIGHT = 1.0 / len(MY_STOCKS)


def initialize(context):
    attach_pipeline(pipe_definition(context), name='my_data')
 
    schedule_function(trade, date_rules.every_day(), time_rules.market_open())
    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())   
    
    context.security = symbol('SPY')
    context.invested = False

def pipe_definition(context):
    universe = Filters.StaticAssets(MY_STOCKS)
    close_price = USEquityPricing.close.latest
    short = Factors.SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=20, mask=universe)   
    long = Factors.SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=60, mask=universe)   
    
    return Pipeline(
            columns = {
            'close_price' : close_price,
            'short' : short,
            'long' : long,
            },
            screen = universe,
            )
    
def before_trading_start(context, data):
    context.output = pipeline_output('my_data')
          
def trade(context, data):
    context.output
     
    open_rules = 'short > long'
    open_these = context.output.query(open_rules).index.tolist()

    for stock in open_these:
        if stock not in context.portfolio.positions and data.can_trade(stock):
            order_target_percent(stock, WEIGHT)
            #Log example
            print('buy :' + str(WEIGHT))           
            cpp = context.portfolio.positions  
            log.info(cpp)  
            cpp_symbols = map(lambda x: x.symbol, cpp)  
            log.info(cpp_symbols)
            
    close_rules = 'short < long'
    close_these = context.output.query(close_rules).index.tolist()
 
    for stock in close_these:
        if stock in context.portfolio.positions and data.can_trade(stock):
            order_target_percent(stock, 0)
            #Log example
            print('sell :' + str(WEIGHT))
            cpp = context.portfolio.positions  
            log.info(cpp)  
            cpp_symbols = map(lambda x: x.symbol, cpp)  
            log.info(cpp_symbols)
            
def record_vars(context, data): 
    record(
        short_mavg = context.output.short,
        long_mavg = context.output.long,
        price = context.output.close_price,
        leverage=context.account.leverage,
        positions=len(context.portfolio.positions)
        )