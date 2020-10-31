#####################################################################
# Trend following algo
# Naoki Nagai, 2015
#####################################################################
# This is a trend following algo for varieties of uncorrelated assets.
# Universe    : Top X% of all securities in terms of trading volume
# Entry signal: Regression line slope exceeds 10% per year & past drawdown and crosses the regression line
# Profit take : 1.96 standard deviation (95% bollinger band)
# Stop loss   : If drawdown exceeds the max drawdown at the time of entry, stop loss. Check every 5 minutes

from numpy import isnan, dot
import numpy as np
import pandas as pd
import statsmodels.api as sm
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume


# Initialization
def initialize(context):
    context.lookback = 252       # Period to calculate slope and draw down
    context.maxlever = 1.0       # Leverage
    context.profittake = 1.96    # 95% bollinger band for profit take
    context.minimumreturn = 0.1  # Entry if annualized slope is over this level
    context.maxdrawdown = 0.10   # Avoid security with too much drawdown
    context.market_impact = 0.2  # Max order is 10% of market trading volume    
    
    context.weights = {}         # Slope at the time of entry.  0 if not to trade
    context.drawdown = {}        # Draw down at the time of entry
    context.shares = {}          # Daily target share
    
    schedule_function(regression, date_rules.every_day(),  time_rules.market_open(minutes=50))
    schedule_function(trade, date_rules.every_day(),  time_rules.market_open(minutes=100))
    schedule_function(trail_stop, date_rules.every_day(),  time_rules.market_open(minutes=30))

    # Run trailing stop and execution every 30 minutes.
    for m in range(1, 391):
        if m % 30 == 0:
            schedule_function(execute, date_rules.every_day(), time_rules.market_open(minutes = m))
            
    # Create and attach our pipeline (top dollar-volume selector), defined below.
    attach_pipeline(high_dollar_volume_pipeline(), 'top_dollar_volume')
        
def high_dollar_volume_pipeline():
    
    # Create a pipeline object. 
    pipe = Pipeline()
    
    # Create a factor for average dollar volume over the last 63 day (1 quarter equivalent).
    dollar_volume = AverageDollarVolume(window_length=63)
    pipe.add(dollar_volume, 'dollar_volume')
    
    # Define high dollar-volume filter to be the top 5% of stocks by dollar volume.
    high_dollar_volume = dollar_volume.percentile_between(95, 100)
    pipe.set_screen(high_dollar_volume)
    
    return pipe

def before_trading_start(context, data):

    context.pipe_output = pipeline_output('top_dollar_volume')
    
    context.security_list = context.pipe_output.index

# Calculate the slopes for different assetes    
def regression(context, data):
    
    prices = data.history(context.security_list, 'open', context.lookback, '1d')

    X=range(len(prices))
        
    # Add column of ones so we get intercept
    A=sm.add_constant(X)
    
    for s in context.security_list:
        # Price movement
        sd = prices[s].std() 

        # Price points to run regression
        Y = prices[s].values
        
        # If all empty, skip
        if isnan(Y).any():
            continue
        
        # Run regression y = ax + b
        results = sm.OLS(Y,A).fit()
        (b, a) =results.params
        
        # a is daily return.  Multiply by 252 to get annualized trend line slope
        slope = a / Y[-1] * 252       # Daily return regression * 1 year

        if slope > 0:
            dd = drawdown(Y)
        
        if slope < 0:
            dd = drawdown(-Y)

        # Currently how far away from regression line?
        delta = Y - (dot(a,X) + b)
        
        # Don't trade if the slope is near flat 
        slope_min = max(dd, context.minimumreturn)   # Max drawdown and minimum return 
        
        # Current gain if trading
        gain = get_gain(context, s)
        
        # Exits
        if s in context.weights and context.weights[s] != 0:
            # Long but slope turns down, then exit
            if context.weights[s] > 0 and slope < 0:
                context.weights[s] = 0
                log.info('v %+2d%% Slope turn bull  %3s - %s' %(gain*100, s.symbol, s.security_name))

            # Short but slope turns upward, then exit
            if context.weights[s] < 0 and 0 < slope:
                context.weights[s] = 0
                log.info('^ %+2d%% Slope turn bear  %3s - %s' %(gain*100, s.symbol, s.security_name))

            # Profit take, reaches the top of 95% bollinger band
            if delta[-1] > context.profittake * sd and s in context.weights and context.weights[s] > 0:
                context.weights[s] = 0        
                log.info('//%+2d%% Long exit %3s - %s'%(gain*100, s.symbol, s.security_name))

            # Profit take, reaches the top of 95% bollinger band
            if delta[-1] < - context.profittake * sd and context.weights[s] < 0:
                context.weights[s] = 0        
                log.info('\\%+2d%% Short exit %3s - %s' %(gain*100, s.symbol, s.security_name))
        # Entry
        else:
            # Trend is up and price crosses the regression line
            if slope > slope_min and delta[-1] > 0 and delta[-2] < 0 and dd < context.maxdrawdown:
                context.weights[s] = slope
                context.drawdown[s] = slope_min
                log.info('/     Long  a = %+.2f%% %3s - %s' %(slope*100, s.symbol, s.security_name),gain)

            # Trend is down and price crosses the regression line
            if slope < -slope_min and delta[-1] < 0 and delta[-2] > 0  and dd < context.maxdrawdown:
                context.weights[s] = slope
                context.drawdown[s] = slope_min
                log.info('\     Short a = %+.2f%% %3s - %s' %(slope*100, s.symbol, s.security_name))

def get_gain(context, s):
    if s in context.portfolio.positions:
        cost =   context.portfolio.positions[s].cost_basis
        amount = context.portfolio.positions[s].amount 
        price =  context.portfolio.positions[s].last_sale_price
        if cost == 0:
            return 0
        if amount > 0:
            gain = price/cost - 1        
        elif amount < 0:
            gain = 1 - price/cost
    else:
        gain = 0
    return gain 


def trade(context, data):
    w = context.weights
    record(leverage_pct = context.account.leverage*100.)
    record(longs = sum(context.portfolio.positions[s].amount > 0 for s in context.portfolio.positions))
    record(shorts = sum(context.portfolio.positions[s].amount < 0 for s in context.portfolio.positions))
    
    positions = sum(w[s] != 0 for s in w)
    held_positions = [p for p in context.portfolio.positions if context.portfolio.positions[p].amount != 0]

    context.securities = context.security_list.tolist() + held_positions
    for s in context.securities:
        if s not in w:
            context.shares.pop(s,0)
            context.drawdown.pop(s,0)
        elif w[s] == 0:
            context.shares.pop(s,0)
            context.drawdown.pop(s,0)
            context.weights.pop(s,0)
        elif w[s] > 0:
            context.shares[s] = context.maxlever/positions 
        elif w[s] < 0:
            context.shares[s] = -context.maxlever/positions

def execute(context,data):
    
    open_orders = get_open_orders()
    
    for s in context.shares:
        if not data.can_trade(s) or s in open_orders:
            continue
        order_target_percent(s, context.shares[s])              
                
# We are entering into position when slope exceeds the drawdown
# If we experience the drawdown again, stop loss
def trail_stop(context, data):
    print get_datetime()
    print 'Positions: %s' % str(context.portfolio.positions.keys())
    prices = data.history(context.portfolio.positions.keys(), 'price', context.lookback, '1d')
    for s in context.portfolio.positions:
        
        if s not in context.weights or context.weights[s] == 0:
            context.shares[s] = 0
            continue

        if s not in prices or s in get_open_orders():
            continue
        
        gain = get_gain(context, s)
        
        if context.portfolio.positions[s].amount > 0:
            if drawdown(prices[s].values) > context.drawdown[s]:
                log.info('x %+2d%% Long  stop loss  %3s - %s' %(gain * 100, s.symbol, s.security_name))
                context.weights[s] = 0
                context.shares[s] = 0

        elif context.portfolio.positions[s].amount < 0:
            if drawdown(-prices[s].values) > context.drawdown[s]:
                log.info('x %+2d%% Short stop loss  %3s - %s' %(gain * 100, s.symbol, s.security_name))
                context.weights[s] = 0
                context.shares[s] = 0


# Reference http://stackoverflow.com/questions/22607324/start-end-and-duration-of-maximum-drawdown-in-python
def drawdown(xs):
    if len(xs) == 0:
        return 0.
    i = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
    if  len(xs[:i]) == 0:
        return 0.
    j = np.argmax(xs[:i]) # start of period
    return abs((xs[i] - xs[j]) / xs[j])
