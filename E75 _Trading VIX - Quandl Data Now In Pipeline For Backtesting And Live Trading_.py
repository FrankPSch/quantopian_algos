# Below is an implementation using VIX Term Structure Following, to determine entry and exit points for
# trading the volatility ETFs VXX (S&P 500 short term futures) and XIV (inverse S&P 500 short term futures). 

# For more information on this strategy check out:

# https://marketsci.wordpress.com/2012/04/11/strategy-1-for-trading-volatility-etps-term-structure-following/
# https://marketsci.wordpress.com/2012/12/20/trading-vxxxiv-by-blindly-following-the-vix-futures-term-structure/
# https://marketsci.wordpress.com/2012/04/13/back-to-basics-y-ax-b/


from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor
from quantopian.pipeline.data.quandl import yahoo_index_vix

import pandas as pd
import numpy as np
from scipy import stats

# Helper funtion for calculating the intercept.
def _intercept(x, y):
    return stats.linregress(x, y)[1]

# Calculate the impact of the term structure (see more about the 'impact' at the link above). Basically,  
# isolate how much of the movement of VXX isn't due to the VIX index. As a handwavy model we'll  
# run a simple regression of the changes in the VIX to the changes in VXX, and use the  
# intercept term as an estimation of the 'impact.'  
class TermStructureImpact(CustomFactor):  
    # Pre-declare inputs and window_length  
    inputs = [yahoo_index_vix.close, USEquityPricing.close]  
    window_length = 20  
    def compute(self, today, assets, out, vix, close):  
        # Get the prices series of just VXX and calculate its daily returns.  
        vxx_returns = pd.DataFrame(close, columns=assets)[sid(38054)].pct_change()[1:]  
        # Since there isn't a ticker for the raw VIX Pipeline feeds us the value of the  
        # VIX for each day in the 'window_length' for each asset. Which kind of makes sense  
        # -- the VIX is the same value for every security.  
        # Since I have a fixed universe I'll just use VXX, one of my securities, to get a single series of  
        # VIX data. You could use any security or integer index to any column, but I'll use one of my  
        # securities just to keep things straight in my head.  
        vix_returns = pd.DataFrame(vix).pct_change()[0].iloc[1:]  
        # Calculate the 'impact.'  
        alpha = _intercept(vix_returns, vxx_returns)  
        out[:] = alpha
        
        
# Put any initialization logic here.  The context object will be passed to
# the other methods in your algorithm.
def initialize(context):
    
    # Set these to zero to evaulate the signal generating ability of the algorithm.
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0)) # FSC for IB

    # Create, register and name a pipeline in initialize.
    pipe = Pipeline()
    attach_pipeline(pipe, 'example')
    
    # Add the TermStrucutreImpact factor to the Pipeline
    ts_impact = TermStructureImpact()    
    pipe.add(ts_impact, 'ts_impact')
    
    # Define our securities, global variables, and schedule when to trade.
    context.vxx = sid(38054)
    context.xiv = sid(40516)
    
    context.impact = 0
    
    schedule_function(allocate, date_rules.every_day(), time_rules.market_open(minutes=30))

    
def before_trading_start(context, data):
    # Pipeline_output returns the constructed dataframe.
    output = pipeline_output('example')
    output = output.dropna()    
    context.impact = output["ts_impact"].loc[context.vxx] # Again, the value of the 'impact' is the same for all securities, but I'll just index on VXX again.
    
    
# Will be called on every trade event for the securities you specify. 
def record_vars(context, data):
    record(lever=context.account.leverage, impact=context.impact)


# Allocate based on whether the 'impact' is positive or negative. 
# If the 'impact' is positive, go long VXX. If the 'impact' is negative, go long XIV.
def allocate(context, data):
    if context.impact > 0:
        if data.can_trade(context.xiv):
            order_target_percent(context.xiv, 0)
        if data.can_trade(context.vxx):
            order_target_percent(context.vxx, 1)
    elif context.impact < 0:
        if data.can_trade(context.vxx):
            order_target_percent(context.vxx, 0)
        if data.can_trade(context.xiv):
            order_target_percent(context.xiv, 1)
    else:
        log.info("Term Structure Impact is Zero")