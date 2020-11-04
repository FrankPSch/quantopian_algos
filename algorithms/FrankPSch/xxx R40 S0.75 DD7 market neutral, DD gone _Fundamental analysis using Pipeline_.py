#
# TonyM_EquityLS_MicheleGrossiNov2017_01
# --------------------------------------
#Based on Michele Grossi's Nov2017 Valuation Ratio algo, adapted to Equity_LongShort, market neutral by TonyM.
#
#=======================================

#Michele Grossi's algo, Nov 2017
# original dates: 1/04/2006 to 11/4/2014
# original account size: $ 100k.

#TonyM edits:
# Set up as Market-Neutral Equity LongShort,
# change cost & slippage & a few other small items.
# Results comparison over the original test period:
# Sharpe ratio increased from 0.62 original to 0.82 now.
# Beta reduced from +0.99 original to -0.06 now.
# MaxDD improved from -50.41% original to -6.73% now.
# This is now a very stable, Quantopian competition-grade algo (just need to increase account size to $10MM) with some scope for your further improvement. Enjoy ;-))

""""
scoring based on valuation ratio
filtered on mkt cap, momentum and volatility
different weight for different ratio
"""

from quantopian.pipeline.data import morningstar

from quantopian.algorithm import attach_pipeline, pipeline_output, order_optimal_portfolio 
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import Q1500US
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage, AverageDollarVolume, Returns, RollingLinearRegressionOfReturns

import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta

import talib
import math
import time

from quantopian.pipeline.data.zacks import EarningsSurprises

import quantopian.optimize as opt

#=================================================================
# Set up as Equity_LongShort market neutral algo, TonyM.
#=================================================================
# Define Constraint Parameter values
#-----------------------------------
# General constraints, as per rules of the Quantcon Singapore 2017 Hackethon & Quantopian Open, whichever is the more stringent.

# Risk Exposures
# --------------
MAX_GROSS_EXPOSURE = 0.90   #NOMINAL leverage = 1.00, but must limit to < 1.10
MAX_BETA_EXPOSURE = 0.05
MAX_SECTOR_EXPOSURE = 0.05
#Dollar Neutral .05
#Position Concentration .10

# Set the Number of positions used
# --------------------------------
NUM_LONG_POSITIONS = 300
NUM_SHORT_POSITIONS = 300

# Maximum position size held for any given stock
# ----------------------------------------------
# Note: the optimizer needs some leeway to operate. If the maximum is too small, the optimizer may be overly constrained.
MAX_SHORT_POSITION_SIZE = 2*1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)
MAX_LONG_POSITION_SIZE = 2*1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)

#=================================================================
     
class Sector(CustomFactor):   
    inputs = [morningstar.asset_classification.       morningstar_sector_code] 
    window_length = 1
    def compute(self, today, assets, out, sector):       
        table = pd.DataFrame(index=assets)
        table ["sector"] = sector[-1]
        out[:] =  table.fillna(0).mean(axis=1)



# Create custom factor #2 Price of 10 days ago.y / Price of 30 days ago.        
class Momentum(CustomFactor):   
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close] 
    window_length = 30
    def compute(self, today, assets, out, close):       
        out[:] = close[-10]/close[0]

        
class Pricetobook(CustomFactor):   
       # Pre-declare inputs and window_length
    inputs = [morningstar.valuation_ratios.pb_ratio] 
    window_length = 1
    def compute(self, today, assets, out, pb):
        table = pd.DataFrame(index=assets)
        table ["pb"] = pb[-1]        
        out[:] = table.fillna(table.max()).mean(axis=1)

        
class Pricetoearnings(CustomFactor):   
       # Pre-declare inputs and window_length
    inputs = [morningstar.valuation_ratios.pe_ratio] 
    window_length = 1
    def compute(self, today, assets, out, pe):
        table = pd.DataFrame(index=assets)
        table ["pe"] = pe[-1]        
        out[:] = table.fillna(table.max()).mean(axis=1)

 

class Roa(CustomFactor):   
    # Pre-declare inputs and window_length
    inputs = [morningstar.operation_ratios.roa] 
    window_length = 1
    def compute(self, today, assets, out, roa):
        table = pd.DataFrame(index=assets)
        table ["roa"] = roa[-1]
        out[:] =  table.fillna(table.min()).mean(axis=1)
        
class Roe(CustomFactor): 
    # Pre-declare inputs and window_length
    inputs = [morningstar.operation_ratios.roe] 
    window_length = 1
    def compute(self, today, assets, out, roe):
        table = pd.DataFrame(index=assets)
        table ["roe"] = roe[-1]
        out[:] =  table.fillna(table.min()).mean(axis=1)
        
class Roic(CustomFactor):   
    # Pre-declare inputs and window_length
    inputs = [morningstar.operation_ratios.roic] 
    window_length = 1
    def compute(self, today, assets, out, roic):
        table = pd.DataFrame(index=assets)
        table ["roic"] = roic[-1]
        out[:] =  table.fillna(table.min()).mean(axis=1)

class Volatility(CustomFactor):  
    # pre-declared inputs and window length  
    inputs = [USEquityPricing.close]  
    window_length = 15 
    # compute standard deviation  
    def compute(self, today, assets, out, close): 
        out[:] = np.std(close, axis=0)         
        
# Create custom factor to calculate a market cap based on yesterday's close
# We'll use this to get the top 2000 stocks by market cap
class MarketCap(CustomFactor):   
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close, morningstar.valuation.shares_outstanding] 
    window_length = 1
    # Compute market cap value
    def compute(self, today, assets, out, close, shares):       
        out[:] = close[-1] * shares[-1]
        
#=================================================================
def make_pipeline():
# Create & return our pipeline (dynamic stock selector). The pipeline is     used to rank stocks based on different factors, including built-in factors, or custom factors. Documentation on pipeline is at:    https://www.quantopian.com/help#pipeline-title
#Break this piece of logic out into its own function to make it easier to test & modify in isolation. In particular, this function can be  copy / pasted into research and run by itself.

# specify momentum, quality, value, and any other factors
# -------------------------------------------------------
    momentum = Momentum()
        # Create and apply a filter representing the top 2000 equities by MarketCap every day
    mkt_cap = MarketCap()
    top_2000 = mkt_cap.top(2000)
    pb = Pricetobook()
    pe = Pricetoearnings()
    roa = Roa()
    roe = Roe()
    roic = Roic()
    volat = Volatility()  # Michelle called this "vol".
    sector = Sector()

    
    # Define universe of securities
    # -----------------------------
    #universe = Q1500US() & price_filter & mkt_cap_MM_filter & liqMM_filter & volat_GT & volat_LT
    universe = top_2000

    # Combined Rank
    # -------------
    # Construct a Factor representing the rank of each asset by our momentum, quality, value, and any other metrics. Aggregate them together here using simple addition. By applying a mask to the rank computations, remove any stocks that failed to meet our initial criteria **BEFORE** computing ranks.  This means that the stock with rank 10.0 is the 10th-lowest stock that was included in the Q1500US.
    
    combined_rank = (
        momentum.rank(mask=universe).zscore() +
        #lower is better
        -1.0*volat.rank(mask=universe).zscore() +
        -1.0*pb.rank(mask=universe).zscore() +
        -1.0*pe.rank(mask=universe).zscore() +
        #higher is better
        3.0*roa.rank(mask=universe).zscore() +
        3.0*roe.rank(mask=universe).zscore() +
        3.0*roic.rank(mask=universe).zscore()     
        )

        



    # Build Filters representing the top & bottom stocks by our combined ranking system. Use these as our tradeable universe each day.
    longs = combined_rank.top(NUM_LONG_POSITIONS)
    shorts = combined_rank.bottom(NUM_SHORT_POSITIONS)

    # Final output of pipeline should only include the top/bottom subset of stocks by our criteria
    long_short_screen = (longs | shorts)
    
    # Define any risk factors that we will want to neutralize. We are chiefly interested in Market Beta as a risk factor. Define it using Bloomberg's beta calculation. Ref: https://www.lib.uwo.ca/business/betasbydatabasebloombergdefinitionofbeta.html
    beta = 0.66*RollingLinearRegressionOfReturns(
                    target=sid(8554),
                    returns_length=5,
                    regression_length=260,
                    mask=long_short_screen
                    ).beta + 0.33*1.0
    

    # Create pipeline
    #----------------
    pipe = Pipeline(columns = {
        'longs':longs,
        'shorts':shorts,
        'combined_rank':combined_rank,
        'top_2000':top_2000,
        'momentum':momentum, 
        'pb':pb,   
        'pe':pe,   
        'roa':roa,
        'roe':roe,
        'roic':roic,
        'sector':sector,
        'volat':volat,
        'market_beta':beta
    },
    screen = long_short_screen)
    return pipe

#=================================================================
#============================================
"""
def initialize(context):
    
    pipe = Pipeline()
    attach_pipeline(pipe, 'ranked_2000')
    
    sector = Sector()
    pipe.add(sector, 'sector')
    
    momentum = Momentum()
    pipe.add(momentum, 'momentum')
    
    pb = Pricetobook()
    pipe.add(pb, 'pb')
    
    pe = Pricetoearnings()
    pipe.add(pe, 'pe')
    
    roa = Roa()
    pipe.add(roa, 'roa')
    
    roe = Roe()
    pipe.add(roe, 'roe')
    
    roic = Roic()
    pipe.add(roic, 'roic')
      
    volat = Volatility()
    pipe.add(volat, 'volat')
        
         
    # Create and apply a filter representing the top 2000 equities by MarketCap every day
    
    mkt_cap = MarketCap()
    top_2000 = mkt_cap.top(2000)
    
    #lower is better
    #---------------
    vol_rank = vol.rank(mask=top_2000, ascending=True)
    pipe.add(vol_rank, 'vol_rank')
           
    pb_rank = pb.rank(mask=top_2000, ascending=True)
    pipe.add(pb_rank, 'pb_rank')
    
    pe_rank = pe.rank(mask=top_2000, ascending=True)
    pipe.add(pe_rank, 'pe_rank')
    
    #higher is better
    #----------------
    roa_rank = roa.rank(mask=top_2000, ascending=False)
    pipe.add(roa_rank, 'roa_rank')
    
    roe_rank = roe.rank(mask=top_2000, ascending=False)
    pipe.add(roe_rank, 'roe_rank')
    
    roic_rank = roic.rank(mask=top_2000, ascending=False)
    pipe.add(roic_rank, 'roic_rank')
    
    #different weight per different ratios 
    #-------------------------------------
    combo_raw = (1*pb_rank+1*pe_rank+3*roa_rank+3*roe_rank+3*roic_rank)/10
    pipe.add(combo_raw, 'combo_raw')
    
    # Rank the combo_raw and add that to the pipeline
    pipe.add(combo_raw.rank(mask=top_2000), 'combo_rank')
    
    #market cap, momentum and volarility filter
    pipe.set_screen(top_2000  & (momentum>1) & (vol_rank.top(400)))
    
  
            
    # Scedule my rebalance function
    schedule_function(func=rebalance, 
date_rule=date_rules.month_start(days_offset=0), 
time_rule=time_rules.market_open(hours=0,minutes=30), half_days=True)
    
    # Schedule my plotting function
    schedule_function(func=record_vars,
date_rule=date_rules.every_day(),
 time_rule=time_rules.market_close(),                      half_days=True)
    
    # set my leverage
    context.long_leverage = 0.95
    
    
            
def before_trading_start(context, data):
    # Call pipelive_output to get the output
    context.output = pipeline_output('ranked_2000')
      
    # Narrow down the securities to only the top 20 & update my universe
    context.long_list = context.output.sort_values(['combo_rank'], ascending=True).iloc[:20]
       

def record_vars(context, data):  
     # Record and plot the leverage of our portfolio over time. 
    record(leverage = context.account.leverage)
    
    print "Long List"
    log.info("\n" + str(context.long_list.sort_values(['combo_rank'], ascending=True).head(3)))
    
  
    
    

# This rebalancing is called according to our schedule_function settings.     
def rebalance(context,data):
    try:
        long_weight = context.long_leverage / float(len(context.long_list))
        
    except ZeroDivisionError:
        long_weight = 0
        
    #maximum weight per single stock
    if long_weight > 0.054 :
        long_weight = 0.05

    for long_stock in context.long_list.index:
        log.info("ordering longs")
        log.info("weight is %s" % (long_weight))
        order_target_percent(long_stock, long_weight)
        
        
    for stock in context.portfolio.positions.iterkeys():
        if stock not in context.long_list.index:
            order_target(stock, 0)
"""            
#================================================================
# Set up as Equity_LongShort market neutral algo, TonyM.
#=================================================================
# Initialization
# --------------
def initialize(context):
#Called once at the start of the algorithm.
    
# Nominal Leverage = Maximum Gross Exposure = 1.00, but re-set this to 0.90 to avoid risk of exceeding hard leverage limit of 1.10
    context.leverage_buffer = 0.90
    
    # Set slippage & commission as per Quantopian Open rules.
    # For competition use, assume $0.001/share
    # Can take up to 2.5% of 1 minute's trade volume.
    set_commission(commission.PerShare(cost=0.001, min_trade_cost=0))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    
    context.spy = sid(8554)
    
    attach_pipeline(make_pipeline(), 'long_short_equity_template')

    # Schedule my rebalance function.
    #-------------------------------
    #Changed from monthly to weekly and then daily rebal, 45 mins after market open.
    schedule_function(func=rebalance,                     date_rule=date_rules.week_start(days_offset=1),                  time_rule=time_rules.market_open(hours=0,minutes=45),                      half_days=True)
    
    # Record tracking variables at the end of each day.
    #-------------------------------------------------
    schedule_function(func=recording_statements,                      date_rule=date_rules.every_day(),                      time_rule=time_rules.market_close(),half_days=True)    
    
    #Initialize a dictionary to store the entry dates for each stock held
    context.entry_dates = dict()
    
#=================================================================
# Control & Monitor Leverage
#---------------------------
def handle_data(context, data):
    # Called every 1 minute bar for securities specified
    pass

#=================================================================
# Record & output my portfolio variables at End of Day only
#----------------------------------------------------------
def recording_statements(context, data):    
    # Track the algorithm's leverage, plot daily on custom graph.
    leverage = context.account.leverage
    record(leverage=leverage)
    
    # warning: 10x delta % leverage vs target leverage of 1.00 (e.g. if leverage = 1.105 --> value displayed = 10*10.5% = 105. Must keep lev < 1.10, which is displayed as 100). Clip to limits of +/- 200 for display. 
    lev1_warn_10xdpct = min(200, max(-200, 10*100*(leverage - 1.00)))    
    record(lev1_warn_10xdpct=lev1_warn_10xdpct)

    num_positions = len(context.portfolio.positions)
    record(num_positions = num_positions)
    
#=================================================================        
def before_trading_start(context, data):
# Called and runs every day before market open. This is where we get the securities that made it through the pipeline and which we are interested in trading each day.
    context.pipeline_data = pipeline_output('long_short_equity_template')

#================================================================= 
# Called at start of each month or week to rebalance Longs & Shorts lists
def rebalance(context, data):
    #my_positions = context.portfolio.positions
    # Optimize API
    pipeline_data = context.pipeline_data

    # Extract from pipeline any specific risk factors to neutralize that have already been calculated 
    risk_factor_exposures = pd.DataFrame({
            'market_beta':pipeline_data.market_beta.fillna(1.0)
        })
    # Fill in any missing factor values with a market beta of 1.0.
    # Do this rather than simply dropping the values because want to err on the side of caution. Don't want to exclude a security just because it is missing a calculated market beta data value, so assume any missing values have full exposure to the market.
 
    # Define objective for the Optimize API. 
    # Here we use MaximizeAlpha because we believe our combined factor ranking to be proportional to expected returns. This routine will optimize the expected return of the algorithm, going long on the highest expected return and short on the lowest.
    
    objective = opt.MaximizeAlpha(pipeline_data.combined_rank)
    
    # Define the list of constraints
    constraints = []
    
    # Constrain maximum gross leverage
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_EXPOSURE))
    
    # Require algorithm to remain dollar-neutral
    constraints.append(opt.DollarNeutral())    # default tolerance = 0.0001
    
    # Add sector neutrality constraint using the sector classifier included in the pipeline
    constraints.append(
        opt.NetGroupExposure.with_equal_bounds(
            labels=pipeline_data.sector,
            min=-MAX_SECTOR_EXPOSURE,
            max=MAX_SECTOR_EXPOSURE,
        ))
    
    # Take the risk factors extracted above and list desired max/min exposures to them. 
    neutralize_risk_factors = opt.FactorExposure(
        loadings=risk_factor_exposures,
        min_exposures={'market_beta':-MAX_BETA_EXPOSURE},
        max_exposures={'market_beta':MAX_BETA_EXPOSURE}
    )
    constraints.append(neutralize_risk_factors)
    
# With this constraint, we enforce that no position can make up greater than MAX_SHORT_POSITION_SIZE on the short side and no greater than MAX_LONG_POSITION_SIZE on the long side. This ensures we don't overly concentrate the portfolio in one security or a small subset of securities.
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))

# Put together all the pieces defined above by passing them into the order_optimal_portfolio function. This handles all ordering logic, assigning appropriate weights to the securities in our universe to maximize alpha with respect to the given constraints.
    order_optimal_portfolio(
        objective=objective,
        constraints=constraints,
    )

#=================================================================
# Python "time test", if required.  Acknowledgement & thanks to Ernesto Perez, Quantopian support.
#start = time.time()
# Block of code you want to test here
#end = time.time()
#log.info(end - start)

#=================================================================