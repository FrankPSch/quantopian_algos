"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.filters import Q1500US 
def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    context.security = symbol('SPY')
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

 
def handle_data(context,data):
    """
    Called every minute.
    """
    print(data)
    MA1 = data[context.security].mavg(8)
    MA2 = data[context.security].mavg(33)
    
    current_price = data[context.security].price
    current_positions =  context.portfolio.positions[symbol('SPY')].amount                                                        
    cash = context.portfolio.cash
    
    if(MA1 > MA2) and current_positions == 0:
        number_of_shares = int(cash/current_price)
        order(context.security, number_of_shares)
        log.info('buying shares')
        
        
    elif (MA1 < MA2) and current_positions != 0:
        order(context.security, -current_positions)
        order_target(context.security, 0)
        log.info('selling shares')
        
    record(MA1 = MA1, MA2 = MA2, price = current_price)