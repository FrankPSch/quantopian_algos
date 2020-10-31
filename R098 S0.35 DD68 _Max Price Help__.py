"""
This algorithm selects the 5 highest dollar volume stocks each day.
It logs the highest price of each every day as long as the stock is held
"""
# The following imports need to included when using Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
import quantopian.optimize as opt

# Import all the built in Quantopian filters and factors (just in case)
import quantopian.pipeline.filters as Filters
import quantopian.pipeline.factors as Factors

# Import Pandas and Numpy (just in case we want to use their functionality)
import pandas as pd
import numpy as np

# Import any specialiazed packages here (eg scipy.optimize or scipy.stats)
pass

# Import any needed datasets
from quantopian.pipeline.data.builtin import USEquityPricing


# Set any 'constants' you will be using
TARGET_QTY = 5

 
def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    # Create a dic to hold "days held" value for each security in our current portfolio
    context.held_days = {}
    
    # Attach the pipeline defined in my_pipe so we have data to use
    attach_pipeline(make_pipeline(), 'my_pipeline')
     
    # Schedule when to log highs
    schedule_function(log_highs, date_rules.every_day(), time_rules.market_close())
    
    # Schedule when to trade
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=1))
     
    # Schedule when to record any variables one wants to watch
    #schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
     
    schedule_function(record_high, date_rules.every_day(), time_rules.market_close())  
    context.day_count = 0  
    context.high_sids  = [ ]  
    context.high_sids_exclude  = [ ]  

    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

def record_high(context, data):  
    context.day_count += 1

    for s in context.portfolio.positions:  
        if not data.can_trade(s): continue  
        if s in context.high_sids_exclude: continue
        if s not in context.held_days: continue
            
        max_high = data.history(s, 'high', context.held_days[s], '1d').max()
        
        # periodically log all  
        if 0 and context.day_count % 126 == 0:  
            log.info('{} {}'.format(s.symbol, int(max_high)))

        # add up to 5 securities for record  
        if len(context.high_sids) < 5 and s not in context.high_sids:  
            context.high_sids.append(s)  
        if s not in context.high_sids: continue     # limit to only them

        # record their profit and loss  
        who  = s.symbol  
        what = data.history(s, 'high', context.held_days[s], '1d').max()  
        record( **{ who: what } )      

def log_highs(context,data):
    """    
    Log the highest high of each currently held security since it was opened
    Note this isn't really exact. It does not include the high on the day it was purchased.
    """
    log.info('holding {} securities'.format(len(context.portfolio.positions)))
    for security, lookback in context.held_days.items():
        highs = data.history(security, 'high', lookback, '1d')
        max_high = highs.max()
        
        log.info('{} had a high of {}'.format(security, max_high))


def make_pipeline():
    '''
    Here is where the pipline definition is set.
    Specifically it defines which collumns appear in the resulting dataframe.
    It can also have a screen which filters which equities (ie rows) are returned
    '''
    
    # Create a universe filter which defines our baseline set of securities
    q500us = Filters.Q500US()
                
    # Create any basic data factors that your logic will use.
    pass

    # Create any built in factors you want to use 
    # Just ensure they are imported first.
    dollar_vol = Factors.AverageDollarVolume(window_length=1, mask=q500us)
    
    # Create any built in filters you want to use.
    pass

    # Create any filters based upon factors defined above.
    # These are easily made with the built in methods such as '.top' etc applied to a factor
    top_dollar_vol = dollar_vol.top(TARGET_QTY)

    # Define the columns and any screen which we want our pipeline to return
    return Pipeline(
            columns = {'dollar_vol' : dollar_vol},
            screen = top_dollar_vol
            )

 
def before_trading_start(context, data):
    '''
    Run pipeline_output to get the latest data for each security.
    The data is returned in a 2D pandas dataframe. Rows are the security objects.
    Columns are what was defined in the pipeline definition.
    '''
    
    # Get a dataframe of our pipe data. Placed in the context object so it's available
    # to other functions and methods (quasi global)
    context.output = pipeline_output('my_pipeline')
    
    # Get a list of the securities that are returned by the pipe.
    context.security_list = context.output.index.tolist()
    
    # Update and increment our 'days_held' dic
    # This keeps track of how long we have held our current holdings
    # We create a new dic with values equal to the current dic + 1
    # Note that the 'get' method returns a 0 value if the security isn't in the current list
    # Set the context.held_days to this new dic. 
    # This avoids having to delete old positions from dic
    held_days = {security: (context.held_days.get(security, 0) + 1) 
                 for security in context.portfolio.positions}

    context.held_days = held_days
    
    
def my_rebalance(context,data):
    """
    Execute orders according to our schedule_function() timing. 
    """
    # First create a series with the weights for each stock
    # This can be done a lot of ways. 
    # Using the plain pandas Series method here
    weight = 1.0 / TARGET_QTY
    weights = pd.Series(index = context.security_list, data = weight)
    
    # Next create a TargetWeights object using our weights series
    target_weights = opt.TargetWeights(weights)
    
    # Finally, execute the order_optimal_portfolio method
    # No need to loop through the stocks. 
    # The order_optimal_portfolio does all the ordering at one time
    order_optimal_portfolio(objective = target_weights, constraints = [])
     
 
def my_record_vars(context, data):
    """
    Plot or log any variables at the end of each day.
    """
    holdings = len(context.portfolio.positions)
    record(holdings=holdings)