from quantopian.pipeline import Pipeline, CustomFactor
from quantopian.algorithm import attach_pipeline, pipeline_output

from quantopian.pipeline.factors import SimpleMovingAverage, RollingLinearRegressionOfReturns
from quantopian.pipeline.filters.morningstar import Q1500US

from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.classifiers.morningstar import Sector

# Alpha Vertex Precog data.
# https://www.quantopian.com/data/alpha_vertex/precog_top_500
from quantopian.pipeline.data.alpha_vertex import precog_top_500 as precog

# EventVestor Earnings Calendar free from 01 Feb 2007 to 1 year ago.
from quantopian.pipeline.factors.eventvestor import (
    BusinessDaysUntilNextEarnings,
    BusinessDaysSincePreviousEarnings,
)

# EventVestor Mergers & Acquisitions free from 01 Feb 2007 to 1 year ago.
from quantopian.pipeline.filters.eventvestor import IsAnnouncedAcqTarget

from quantopian.pipeline.factors import BusinessDaysSincePreviousEvent

import quantopian.optimize as opt

import numpy as np
import pandas as pd

# Constraint Parameters
MAX_GROSS_LEVERAGE = 1.0
MAX_SHORT_POSITION_SIZE = 0.1  # 10%
MAX_LONG_POSITION_SIZE = 0.1   # 10%

# Risk Exposures
MAX_SECTOR_EXPOSURE = 0.10
MAX_BETA_EXPOSURE = 0.20
 
def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    # Rebalance every day, 1 hour after market open.
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open())
     
    # Record tracking variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
     
    # Create our dynamic stock selector.
    attach_pipeline(make_pipeline(), 'my_pipeline')
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    

class PredictionQuality(CustomFactor):
    """
    Create a customized factor to calculate the prediction quality
    for each stock in the universe.
    
    Compares the percentage of predictions with the correct sign 
    over a rolling window (3 weeks) for each stock.
   
    """

    # data used to create custom factor
    inputs = [precog.predicted_five_day_log_return, USEquityPricing.close]

    # change this to what you want
    window_length = 15

    def compute(self, today, assets, out, pred_ret, px_close):

        log_ret5 = np.log(px_close) - np.log(np.roll(px_close, 5, axis=0))

        log_ret5 = log_ret5[5:]
        n = len(log_ret5)
        
        # predicted returns
        pred_ret = pred_ret[:n]

        # number of predictions with incorrect sign
        err = np.absolute((np.sign(log_ret5) - np.sign(pred_ret)))/2.0

        # custom quality measure
        pred_quality = (1 - pd.DataFrame(err).ewm(min_periods=n, com=n).mean()).iloc[-1].values
        
        out[:] = pred_quality
        
    
def make_pipeline():
    """
    Dynamically apply the custom factors defined below to 
    select candidate stocks from the PreCog universe 
    
    """
    
    pred_quality_thresh      = 0.5
    
     # Filter for stocks that are not within 2 days of an earnings announcement.
    not_near_earnings_announcement = ~((BusinessDaysUntilNextEarnings() <= 2)
                                | (BusinessDaysSincePreviousEarnings() <= 2))
    
    # Filter for stocks that are announced acquisition target.
    not_announced_acq_target = ~IsAnnouncedAcqTarget()
    
    # Our universe is made up of stocks that have a non-null sentiment & precog signal that was 
    # updated in the last day, are not within 2 days of an earnings announcement, are not announced 
    # acquisition targets, and are in the Q1500US.
    universe = (
        Q1500US() 
        & precog.predicted_five_day_log_return.latest.notnull()
        & not_near_earnings_announcement
        & not_announced_acq_target
    )
 
    # Prediction quality factor.
    prediction_quality = PredictionQuality(mask=universe)
    
    # Filter for stocks above the threshold quality.
    quality= prediction_quality > pred_quality_thresh

    latest_prediction = precog.predicted_five_day_log_return.latest
    
    non_outliers = latest_prediction.percentile_between(1,99, mask=quality)
    normalized_return = latest_prediction.zscore(mask=non_outliers)
    
    normalized_prediction_rank = normalized_return.rank()
    
    prediction_rank_quantiles = normalized_prediction_rank.quantiles(5)
    
    longs = prediction_rank_quantiles.eq(4)
    shorts = prediction_rank_quantiles.eq(0)
    
    # We will take market beta into consideration when placing orders in our algorithm.
    beta = RollingLinearRegressionOfReturns(
                    target=sid(8554),
                    returns_length=5,
                    regression_length=260,
                    mask=(longs | shorts)
    ).beta
    
    # We will actually be using the beta computed using Bloomberg's computation.
    # Ref: https://www.lib.uwo.ca/business/betasbydatabasebloombergdefinitionofbeta.html
    bb_beta = (0.66 * beta) + (0.33 * 1.0)

    ## create pipeline
    columns = {
        'longs': longs,
        'shorts': shorts,
        'market_beta': bb_beta,
        'sector': Sector(),
    }
    pipe = Pipeline(columns=columns, screen=(longs | shorts))
 
    return pipe
 
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.pipeline_output = pipeline_output('my_pipeline')
  
    # These are the securities that we are interested in trading each day.
    context.security_list = context.pipeline_output.index.tolist()
    
    # Replace NaN beta values with 1.0.
    context.market_beta = context.pipeline_output.market_beta.fillna(1.0)
    
 
def my_rebalance(context,data):
    """
    Place orders according to our schedule_function() timing.
    """
    
    # Compute our portfolio weights.
    long_secs = context.pipeline_output[context.pipeline_output['longs']].index
    long_weight = 0.5 / len(long_secs)
    
    short_secs = context.pipeline_output[context.pipeline_output['shorts']].index
    short_weight = -0.5 / len(short_secs)
    
    todays_weights = {}
    
    # Open our long positions.
    for security in long_secs:
        todays_weights[security] = long_weight
    
    # Open our short positions.
    for security in short_secs:
        todays_weights[security] = short_weight
    
    # Sets our objective to maximize alpha based on the weights we receive from our factor.
    objective = opt.MaximizeAlpha(todays_weights)

    # Constraints
    # -----------
    # Constrain our gross leverage to 1.0 or less. This means that the absolute
    # value of our long and short positions should not exceed the value of our
    # portfolio.
    constrain_gross_leverage = opt.MaxGrossExposure(MAX_GROSS_LEVERAGE)
    
    # Constrain individual position size to no more than a fixed percentage 
    # of our portfolio. Because our alphas are so widely distributed, we 
    # should expect to end up hitting this max for every stock in our universe.
    constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
        -MAX_SHORT_POSITION_SIZE,
        MAX_LONG_POSITION_SIZE,
    )

    # Constrain ourselves to allocate the same amount of capital to 
    # long and short positions.
    dollar_neutral = opt.DollarNeutral(tolerance=0.05)
    
    # Market neutrality constraint. Our portfolio should not be over-exposed
    # to the performance of the market (beta-to-spy).
    market_neutral = opt.FactorExposure(
        loadings=pd.DataFrame({'market_beta': context.market_beta}),
        min_exposures={'market_beta': -MAX_BETA_EXPOSURE},
        max_exposures={'market_beta': MAX_BETA_EXPOSURE},
    )
    
    # Sector neutrality constraint. Our portfolio should not be over-
    # exposed to any particular sector.
    sector_neutral = opt.NetGroupExposure.with_equal_bounds(
            labels=context.pipeline_output.sector,
            min=-MAX_SECTOR_EXPOSURE,
            max=MAX_SECTOR_EXPOSURE,
    )
    
    order_optimal_portfolio(
        objective=objective,
        constraints=[
            constrain_gross_leverage,
            constrain_pos_size,
            dollar_neutral,
            market_neutral,
            sector_neutral,
        ],
        #universe=context.security_list,
    )
    
 
def my_record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    long_count = 0
    short_count = 0

    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            long_count += 1
        if position.amount < 0:
            short_count += 1
            
    # Plot the counts
    record(num_long=long_count, num_short=short_count, leverage=context.account.leverage)