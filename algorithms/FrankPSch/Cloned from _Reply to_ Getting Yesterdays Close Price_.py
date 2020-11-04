"""
This is a sample algorithm on Quantopian showing the use of previous close price.
"""

# Import necessary Pipeline modules
from quantopian.pipeline import Pipeline, CustomFactor
from quantopian.algorithm import attach_pipeline, pipeline_output

# Import specific filters and factors which will be used
from quantopian.pipeline.filters import QTradableStocksUS, StaticAssets
from quantopian.pipeline.factors import Returns

# Import datasets which will be used
from quantopian.pipeline.data.builtin import USEquityPricing

# import optimize
import quantopian.optimize as opt
 
# Import pandas
import pandas as pd

def initialize(context):
    """
    Initialize constants, create pipeline, and schedule functions
    This uses the default slippage and commission models
    """
    # Universe we wish to trade
    # Place one or more desired symbols below
    context.MY_STOCKS = symbols('SPY', 'GLD')
    
    # Create our weights (evenly weighted)
    context.WEIGHT = 1.0 / len(context.MY_STOCKS)

    # Make our pipeline and attach to the algo
    attach_pipeline(make_pipeline(context), 'my_pipe')

    # Place orders
    schedule_function(
        func=place_orders_using_optimize,
        date_rule=date_rules.week_start(),
        time_rule=time_rules.market_open()
    )
    

def make_pipeline(context):
    """
    Define a pipeline.
    """
    # Specify the universe of securities
    base_universe = StaticAssets(context.MY_STOCKS)
    
    # Create any needed factors.
    close_price = USEquityPricing.close.latest
    
    close_1_yr_ago_factor = Factor_N_Days_Ago(
        inputs=[USEquityPricing.close], 
        window_length=252)
    
    close_1_yr_ago_calculated = close_price / (Returns(window_length=252) + 1.0)
        
    # Create our pipeline
    return Pipeline(
      columns={
        'close_price': close_price,
        'close_1_yr_ago_factor': close_1_yr_ago_factor,
        'close_1_yr_ago_calculated': close_1_yr_ago_calculated,
         },
      screen=base_universe
    )
  
        
def before_trading_start(context, data):
    """
    Run our pipeline to fetch the actual data. 
    It's a good practice to place the pipeline execution here. 
    This gets allocated more time than scheduled functions.
    """
    context.output = pipeline_output('my_pipe')
    
    log.info(context.output)
    
    
def place_orders_using_optimize(context, data):
    """
    Use Optimize to place orders all at once
    """
    # Make a series of the securities to order and associated weights
    # Ensure that all the short weights are negative (this is what tells opt to short them)
    weights = pd.Series(context.WEIGHT, context.output.index)

    # Create our TargetWeights objective
    target_weights = opt.TargetWeights(weights) 

    # Execute the order_optimal_portfolio method with above objective and constraint
    # No need to loop through the stocks. 
    # The order_optimal_portfolio does all the ordering at one time
    # Also closes any positions not in 'weights'
    # As a bonus also checks for 'can_trade'
    # Could set constraints here if desired
    order_optimal_portfolio(
        objective = target_weights,
        constraints = []
    )
    
######## CUSTOM FACTORS ###########

class Factor_N_Days_Ago(CustomFactor):
    """
    Returns the factor value N days ago where window_length=N
    This is the price adjusted as of the current simulation day.
    """
    def compute(self, today, assets, out, close_price): 
        out[:] = close_price[0]