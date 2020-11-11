'''
This algorithm takes the top 2000 companies by market cap (the Russell 2000) and ranks them by their 10 day returns. It longs in the top 10% and shorts the bottom 10% and rebalances monthly. 
'''

from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline import CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar


# Create custom factor to calculate the returns
class Returns(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close] 
    
    # Compute factor1 value
    def compute(self, today, assets, out, close):       
        out[:] = 1-((close[0]-close[-1])/close[-1])
        
# Create custom factor to calculate a market cap based on yesterday's close
class MarketCap(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close, morningstar.valuation.shares_outstanding] 
    window_length = 1
    
    # Compute market cap value
    def compute(self, today, assets, out, close, shares):       
        out[:] = close[-1] * shares[-1]

def initialize(context):
    
    context.long_leverage = 0.50
    context.short_leverage = -0.50
    
    pipe = Pipeline()
    attach_pipeline(pipe, 'ranked_2000')

    # Get the share class using .latest to get the most recent value
    share_class = morningstar.share_class_reference.is_primary_share.latest    
    
    # Create and apply a filter representing the top 2000 equities by MarketCap every day
    # Mask out non primary share classes
    # This is an approximation of the Russell 2000
    mkt_cap = MarketCap()
    top_2000 = mkt_cap.top(2000, mask=share_class.eq(1))
    pipe.set_screen(top_2000)

    # add a 10 day returns factor to the pipeline
    # note we are declaring the window_length here becase no default was declared
    returns_10 = Returns(window_length=10)
    pipe.add(returns_10, 'returns_10')
    # add a factor to rank the returns, mask this with the top_2000 filters
    pipe.add(returns_10.rank(mask=top_2000, ascending=False), 'returns_10_rank')
    
    # Scedule my rebalance function
    schedule_function(func=rebalance, 
                      date_rule=date_rules.month_start(days_offset=0), 
                      time_rule=time_rules.market_open(hours=0,minutes=30), 
                      half_days=True)
    
            
def before_trading_start(context, data):
    # Call pipelive_output to get the output
    # Note this is a dataframe where the index is the SIDs for all securities to pass my screen
    # and the colums are the factors which I added to the pipeline
    context.output = pipeline_output('ranked_2000')
    log.info(len(context.output))
      
    # Narrow down the securities to only the top 500 & update my universe
    context.long_list = context.output.sort(['returns_10_rank'], ascending=True).iloc[:200]
    context.short_list = context.output.sort(['returns_10_rank'], ascending=True).iloc[-200:]   
    
    update_universe(context.long_list.index.union(context.short_list.index)) 


def handle_data(context, data):  
    
     # Record and plot the leverage of our portfolio over time. 
    record(leverage = context.account.leverage)
    
    print "Long List"
    log.info("\n" + str(context.long_list.sort(['returns_10_rank'], ascending=True).head(10)))
    
    print "Short List" 
    log.info("\n" + str(context.short_list.sort(['returns_10_rank'], ascending=True).head(10)))

# This rebalancing is called according to our schedule_function settings.     
def rebalance(context,data):
    
    long_weight = context.long_leverage / float(len(context.long_list))
    short_weight = context.short_leverage / float(len(context.short_list))

    
    for long_stock in context.long_list.index:
        if long_stock in data:
            order_target_percent(long_stock, long_weight)
        
    for short_stock in context.short_list.index:
        if short_stock in data:
            order_target_percent(short_stock, short_weight)
        
    for stock in context.portfolio.positions.iterkeys():
        if stock not in context.long_list.index and stock not in context.short_list.index:
            order_target(stock, 0)
