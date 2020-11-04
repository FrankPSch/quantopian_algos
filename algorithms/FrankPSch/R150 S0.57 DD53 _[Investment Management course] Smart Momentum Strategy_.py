'''
For algorithm explanations and sources, please see the report
Pipeline code is based on Quantopian Tutorial 2: Pipeline
Trading code is based on Quantopian Sample Mean Reversion Algorithm
'''

from quantopian.algorithm import order_optimal_portfolio
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
import quantopian.optimize as opt
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import Returns, MarketCap
from quantopian.pipeline.filters.fundamentals import Q1500US

def initialize(context):
    #Start rebalancing procedure at the end of each month, at market open
    schedule_function(
        my_rebalance,
        date_rules.month_end(),
        time_rules.market_open()
    )

    #Record variables at the end of each day.
    schedule_function(
        my_record_vars,
        date_rules.every_day(),
        time_rules.market_close()
    )

    # Create our pipeline and attach it to our algorithm.
    my_pipe = make_pipeline()
    attach_pipeline(my_pipe, 'my_pipeline')

def make_pipeline():
    #0. Select top-1500 liquid stocks
    base_universe = Q1500US()
    
    #1. Market Capitalization Filter, select top 1200 out of 1500
    market_cap = MarketCap(mask=base_universe)
    top_market_cap = market_cap.top(1200)
    
    #2. B/M Filter, select top 5000 out of all stocks (~8200)
    btom = morningstar.valuation_ratios.book_value_yield.latest
    top_btom = btom.top(5000)
    
    #3. Interception of 1st and 2nd filters results, on average returns ~1000 stocks
    top_cap_btom = top_market_cap & top_btom
    
    #4. Top Performers filter, select top 100
    latest_return_7_1 = Returns(
        inputs=[USEquityPricing.close],
        window_length=147,
        mask=top_cap_btom
      )/Returns(
        inputs=[USEquityPricing.close],
        window_length=21,
        mask=top_cap_btom
        )
    top_performers = latest_return_7_1.top(100)
    
    #5. Smoothest Returns filter, select 50 smoothest
    r_7 = Returns(window_length = 21, mask=top_performers)
    r_6 = (Returns(window_length = 42, mask=top_performers)+1)/(r_7+1)-1
    r_5 = (Returns(window_length = 63, mask=top_performers)+1)/((r_7+1)*(r_6+1))-1
    r_4 = (Returns(window_length = 84, mask=top_performers)+1)/((r_7+1)*(r_6+1)*(r_5+1))-1
    r_3 = (Returns(window_length = 105, mask=top_performers)+1)/((r_7+1)*(r_6+1)*(r_5+1)*(r_4+1))-1
    r_2 = (Returns(window_length = 126, mask=top_performers)+1)/((r_7+1)*(r_6+1)*(r_5+1)*(r_4+1)*(r_3+1))-1
    r_1 = (Returns(window_length = 147, mask=top_performers)+1)/((r_7+1)*(r_6+1)*(r_5+1)*(r_4+1)*(r_3+1)*(r_2+1))-1
    r_mean = (r_1+r_2+r_3+r_4+r_5+r_6)/6.0
    varr = ((r_1-r_mean) ** 2+(r_2-r_mean) ** 2+(r_3-r_mean) ** 2+(r_4-r_mean) ** 2+(r_5-r_mean) ** 2+(r_6-r_mean) ** 2)/6.0 #Sample variance of monthly returns of each stock
    top_smooth = varr.bottom(50)
    
    #6. Screening filter, if (tradeable) then buy this stock
    tradeable = top_smooth
    
    return Pipeline(
        screen=tradeable,
        columns={
            'Tradeable': tradeable
        }
    )
def compute_target_weights(context, data): #Computes weights on each stock we are intrested in
    # Initialize empty target weights dictionary.
    # This will map securities to their target weight.
    weights = {}
    
    #Set weights to (tradeable) stocks to create equally-weighted portfolio 
    if context.tradeable:
        long_weight = 1.0 / len(context.tradeable)
    else:
        return weights
    
    # Exit positions in our portfolio if they are not (tradeable)
    for security in context.portfolio.positions:
        if security not in context.tradeable and data.can_trade(security):
            weights[security] = 0

    for security in context.tradeable:
        weights[security] = long_weight

    return weights

def before_trading_start(context, data):
    # Gets our pipeline output every day.
    pipe_results = pipeline_output('my_pipeline')

    # Go long in (tradeable) stocks
    context.tradeable = []
    for sec in pipe_results[pipe_results['Tradeable']].index.tolist():
        if data.can_trade(sec):
            context.tradeable.append(sec)

def my_rebalance(context, data): #Rebalances the portfolio
    # Calculate target weights to rebalance
    target_weights = compute_target_weights(context, data)
    
    # If we have target weights, rebalance our portfolio
    # Rebalance portfolio in the end of each month except January
    today = get_datetime()
    if today.month != 1:
        if target_weights:
            order_optimal_portfolio(
                objective=opt.TargetWeights(target_weights),
                constraints=[],
            )

def my_record_vars(context, data): #Tracks number of long positions
    longs = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1

    #Record our variables.
    record(
        long_count=longs
    )