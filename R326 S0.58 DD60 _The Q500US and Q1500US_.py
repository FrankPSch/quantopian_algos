"""
This is a template algorithm on Quantopian for you to adapt and fill in.  
"""
from quantopian.algorithm import attach_pipeline, pipeline_output  
from quantopian.pipeline import Pipeline  
from quantopian.pipeline.data.builtin import USEquityPricing  
from quantopian.pipeline.factors import AverageDollarVolume  
from quantopian.pipeline.filters import Q1500US  
def initialize(context):  
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    """  
    Called once at the start of the algorithm.  
    """  
    # Rebalance every day, 1 hour after market open.  
    schedule_function(my_rebalance, date_rules.week_start(days_offset=1), time_rules.market_open(hours=1))  
    # Record tracking variables at the end of each day.  
    # schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())  
    # Create our dynamic stock selector.  
    attach_pipeline(make_pipeline(), 'my_pipeline')  

    
def make_pipeline():  
    """  
    A function to create our dynamic stock selector (pipeline). Documentation on  
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title  
    """  

    # Create a dollar volume factor.  
    dollar_volume = AverageDollarVolume(window_length=1)  
    # Pick the top 1% of stocks ranked by dollar volume.  
    high_dollar_volume = dollar_volume.percentile_between(99, 100)  
    pipe = Pipeline(  
        # screen = (high_dollar_volume & Q500US()),  
        screen = Q1500US(),  
        columns = {  
            'dollar_volume': dollar_volume  
        }  
    )  
    return pipe  


def before_trading_start(context, data):  
    """  
    Called every day before market open.  
    """  
    context.output = pipeline_output('my_pipeline')  
    # These are the securities that we are interested in trading each day.  
    context.security_list = context.output.index  
    
    
def my_assign_weights(context, data):  
    """  
    Assign weights to securities that we want to order.  
    """  
    pass  


def my_rebalance(context,data):  
    """  
    Execute orders according to our schedule_function() timing.  
    """  
    # pass  
    weight = 1.0/len(context.security_list)  
    for stock in context.security_list:  
        if data.can_trade(stock):  
            order_target_percent(stock,weight)  
    for stock in context.portfolio.positions.keys():  
        if stock not in context.security_list:  
            if data.can_trade(stock):  
                order_target_percent(stock,0)  
                
                
def my_record_vars(context, data):  
    """  
    Plot variables at the end of each day.  
    """  
    pass