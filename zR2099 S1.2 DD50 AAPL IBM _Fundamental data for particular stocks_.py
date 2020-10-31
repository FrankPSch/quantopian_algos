'''
Simple algorithm to maintain equal dollar value of securities.
Algorithm holds long positions only.
Security universe is fixed.
Selection within universe is based upon meeting fundamental criteria.
'''

# import pipeline methods 
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
import quantopian.optimize as opt

# import built in factors and filters
import quantopian.pipeline.factors as Factors
import quantopian.pipeline.filters as Filters

# import any datasets we need
from quantopian.pipeline.data.builtin import USEquityPricing 
from quantopian.pipeline.data import Fundamentals

# import numpy and pandas just in case
import numpy as np
import pandas as pd


# Here we specify the securities 'universe' to trade

SECURITY_UNIVERSE = symbols(
                        'AAPL',
                        'IBM',
                        'AMZN',
                        'NFLX',
                        )


def initialize(context):
    """
    Called once at the start of the algorithm.
    """   

    # Set commision model
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0))

    # Ensure no short trading (just a precaution)
    set_long_only()
    
    # Create and attach pipeline to get data
    attach_pipeline(my_pipeline(context), name='my_pipeline')
    
    
    # Place orders once a day
    schedule_function(enter_trades, date_rules.every_day(), time_rules.market_open())
    
    # Record tracking variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())

    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

    
def my_pipeline(context):
    '''
    Define the pipline data columns
    '''
    
    # Create filter for just the securities we want to trade
    universe = Filters.StaticAssets(SECURITY_UNIVERSE)

    # Create the fundamental factors we want to use in our trade decisions
    pb_ratio = Fundamentals.pb_ratio.latest
    pe_ratio = Fundamentals.pe_ratio.latest


    # Define our pipeline
    return Pipeline(
            columns = {
            'pb_ratio' : pb_ratio,
            'pe_ratio' : pe_ratio,
            },
            screen = universe,
            )


def before_trading_start(context, data):
    
    # Get the data
    context.output = pipeline_output('my_pipeline')

     
def enter_trades(context, data):

    # First determine which securities we want to be holding
    # Place selection logic in the query method
    securities_to_hold = context.output.query('pe_ratio > 1.0').index

    # Next create a series with the weights for each stock
    # Here we assume equal weight for each stock
    weight = 1.0 / len(securities_to_hold)
    weights = pd.Series(index = securities_to_hold, data = weight)
    
    # Next create a TargetWeights object using our weights series
    # This will become our ordering objective 
    target_weights = opt.TargetWeights(weights)
    
    # Finally, execute the order_optimal_portfolio method
    # No need to loop through the stocks. 
    # The order_optimal_portfolio does all the ordering at one time
    # Also closes any positions not in 'securities_to_hold'
    # We also don't impose any constraints. Just adjust to desired weights
    order_optimal_portfolio(objective = target_weights, constraints = [])


def my_record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
            
    record(securities = len(context.portfolio.positions))