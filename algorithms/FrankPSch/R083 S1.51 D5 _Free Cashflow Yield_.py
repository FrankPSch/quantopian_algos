"""
Trading Strategy using Fundamental Data
1. Look at stocks in the Q1500US.
2. Go long in the top decile of stocks by FCF yield
3. Go short in the bottom decile of stocks by FCF yield.
4. Rebalance weekly at market open.
"""

from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.filters import Q1500US


def initialize(context):
    
    # Rebalance weekly at open
    schedule_function(rebalance,date_rule=date_rules.week_start(),time_rule=time_rules.market_open())
    #Record end of day positions
    schedule_function(my_record_vars, date_rules.week_start(), time_rules.market_close())
    attach_pipeline(make_pipeline(), 'fundamentals_pipeline')

def make_pipeline():
    #Get free cash flow yield from morningstart database (FCF/price)
    fcf = Fundamentals.fcf_yield.latest.rank(mask = Q1500US(), method ='average')
    universe = (
        Q1500US() 
        & fcf.notnull() 
        #& is_fresh
        # & volatility_filter
    )
    #Sort into deciles
    num_quantiles = 5
    fcf_quantiles = fcf.quantiles(num_quantiles)
    #Build Pipeline, long top, short bottom decile
    pipe = Pipeline(screen=universe, columns = {'FCF': fcf, 'longs': fcf_quantiles.eq(num_quantiles-1), 'shorts': fcf_quantiles.eq(0)})

    return pipe
    
"""
Runs our fundamentals pipeline before the market opens every week
"""
def before_trading_start(context, data): 

    context.pipe_output = pipeline_output('fundamentals_pipeline')

    #Long top decile of FCF yield
    context.longs = context.pipe_output[context.pipe_output['longs']].index

    # Short bottom decile of FCF yield
    context.shorts = context.pipe_output[context.pipe_output['shorts']].index

def rebalance(context, data):
    
    my_positions = context.portfolio.positions
    

    if (len(context.longs) > 0) and (len(context.shorts) > 0):

        # Equally weight all of our long positions and all of our short positions.
        long_weight = 0.5/len(context.longs)
        short_weight = -0.5/len(context.shorts)
        
        # Get our target names for our long and short baskets. We can display these
        # later.
        target_long_symbols = [s.symbol for s in context.longs]
        target_short_symbols = [s.symbol for s in context.shorts]

        log.info("Opening long positions each worth %.2f of our portfolio in: %s" \
                 % (long_weight, ','.join(target_long_symbols)))
        
        log.info("Opening long positions each worth %.2f of our portfolio in: %s" \
                 % (short_weight, ','.join(target_short_symbols)))
        
        # Open long positions in our high p/e stocks.
        for security in context.longs:
            if data.can_trade(security):
                if security not in my_positions:
                    order_target_percent(security, long_weight)
            else:
                log.info("Didn't open long position in %s" % security)

        # Open short positions in our low p/e stocks.
        for security in context.shorts:
            if data.can_trade(security):
                if security not in my_positions:
                    order_target_percent(security, short_weight)
            else:
                log.info("Didn't open short position in %s" % security)
                  

    closed_positions = []
    
    # Close our previous positions that are no longer in our pipeline.
    for security in my_positions:
        if security not in context.longs and security not in context.shorts \
        and data.can_trade(security):
            order_target_percent(security, 0)
            closed_positions.append(security)
    
    log.info("Closing our positions in %s." % ','.join([s.symbol for s in closed_positions]))

def my_record_vars(context, data):
    """
    Record variables at the end of each rebalancing period.
    """
    longs = shorts = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
        elif position.amount < 0:
            shorts += 1
    # Record our variables.
    record(leverage=context.account.leverage, long_count=longs, short_count=shorts)