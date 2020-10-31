"""
Utilizes the filters for a good trading universe laid out by Scott Sanderson:
https://www.quantopian.com/posts/string-columns-now-available-in-pipeline
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import morningstar as mstar
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume, SimpleMovingAverage
from quantopian.pipeline.filters.morningstar import IsPrimaryShare
from quantopian.pipeline.filters.morningstar import Q1500US

def initialize(context):
    # Equity numbers for the mean reversion algorithm.
    context.num_securities = 20
    context.num_short = context.num_securities // 2
    context.num_long = context.num_securities - context.num_short
    
    schedule_function(my_rebalance, date_rules.week_start(), time_rules.market_open(hours=1))
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
    
    attach_pipeline(my_pipeline(context), 'my_pipeline')
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB


def my_pipeline(context):
    pipe = Pipeline()
    """
    The original algorithm used the following filters:
        1. common stock
        2 & 3. not limited partnership - name and database check
        4. database has fundamental data
        5. not over the counter
        6. not when issued
        7. not depository receipts
        8. primary share
        9. high dollar volume
    Check Scott's notebook for more details.
    
    This updated version uses Q1500US, one of the pipeline's built-in base universes. 
    Lesson 11 from the Pipeline tutorial offers a great overview of using multiple 
    filters vs using the built-in base universes:
    https://www.quantopian.com/tutorials/pipeline#lesson11
    
    More detail on the selection criteria of this filter can be found  here:
    https://www.quantopian.com/posts/the-q500us-and-q1500us 
    """
    base_universe = Q1500US()
    
    # The example algorithm - mean reversion. Note the tradable filter used as a mask.
    sma_10 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=10,
                                 mask=base_universe)
    sma_30 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=30,
                                 mask=base_universe)
    rel_diff = (sma_10 - sma_30) / sma_30
    
    top_rel_diff = rel_diff.top(context.num_short)
    pipe.add(top_rel_diff, 'top_rel_diff')
    bottom_rel_diff = rel_diff.bottom(context.num_long)
    pipe.add(bottom_rel_diff, 'bottom_rel_diff')
     
    return pipe

# Get the pipeline output and specify which equities to trade.
def before_trading_start(context, data):
    context.output = pipeline_output('my_pipeline')
    context.short_set = set(context.output[context.output['top_rel_diff']].index)
    context.long_set = set(context.output[context.output['bottom_rel_diff']].index)
    context.security_set = context.long_set.union(context.short_set)

# Rebalance weekly.
def my_rebalance(context,data):
    for stock in context.security_set:
        if data.can_trade(stock):
            if stock in context.long_set:
                order_target_percent(stock, 1. / context.num_securities)
            elif stock in context.short_set:
                order_target_percent(stock, -1. / context.num_securities)
        
    for stock in context.portfolio.positions:
        if stock not in context.security_set and data.can_trade(stock):
            order_target_percent(stock, 0)

# Record variables.
def my_record_vars(context, data):
    shorts = longs = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount < 0:
            shorts += 1
        elif position.amount > 0:
            longs += 1
    record(leverage=context.account.leverage, short_count=shorts, long_count=longs)