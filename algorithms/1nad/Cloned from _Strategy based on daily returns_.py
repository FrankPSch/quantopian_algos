from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import SimpleMovingAverage, AverageDollarVolume
from quantopian.pipeline import CustomFactor
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.filters import Q1500US, Q500US
import numpy as np
import pandas as pd
from datetime import date, timedelta


class Daily(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.open,USEquityPricing.close] 
    window_length = 1
    
    # Compute factor1 value
    def compute(self, today, assets, out, open, close):
        out[:] = close[-1,:] / open[-1,:]
        
class AvgDailyDollarVolumeTraded(CustomFactor):
    
    inputs = [USEquityPricing.close, USEquityPricing.volume]
    window_length = 40    
    def compute(self, today, assets, out, close_price, volume):
        out[:] = np.mean(close_price * volume, axis=0)        
      
class Previous_night(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.open,USEquityPricing.close] 
    window_length = 1
    # Compute factor1 value
    def compute(self, today, assets, out, open, close):
        out[:] = (open[-1] / close[0]) 
      
    # Create custom factor #2         

  
class Market(CustomFactor):   
    
    # Pre-declare market cap of every security on the day.
    inputs = [morningstar.valuation.market_cap] 
    window_length = 1
    
    # Compute factor2 value
    def compute(self, today, assets, out, inputs):
        out[:] = inputs[-1]
class Liquidity(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.volume, morningstar.valuation.shares_outstanding] 
    window_length = 1
    
    # Compute factor2 value
    def compute(self, today, assets, out, volume, shares):       
        out[:] = volume[-1]/shares[-1] 

class Momentum(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.open]
    window_length = 252
    def compute(self, today, assets, out, close, open):
        out[:] = close[-21]/close[0]
                    
def initialize(context):
    pipe = Pipeline()
    attach_pipeline(pipe, 'ranked_2000')
       
    # Add the two factors defined to the pipeline
    daily = Daily()
    pipe.add(daily, 'daily')    
  
    market = Market()
    pipe.add(market, 'market')
    
    liquid= Liquidity()
    pipe.add(liquid, 'liquid')
    
    momentum= Momentum()
    pipe.add(momentum,'momentum')
    
    previous=Previous_night()
    pipe.add(previous,'previous')
    
    # Create and apply a filter representing the Q1500US
    top_500 = Q1500US() 

    # Rank factor 1 and add the rank to our pipeline
    daily_rank = daily.rank(mask=top_500)
    pipe.add(daily_rank, 'daily_rank')
              
    previous_rank=previous.rank(mask=top_500,ascending=False)
    pipe.add(previous_rank,'previous_rank')
    # Rank factor 2 and add the rank to our pipeline
    market_rank = market.rank(mask=top_500,ascending =False)
    pipe.add(market_rank, 'market_rank')
    
    liquid_rank=liquid.rank(mask=top_500)
    pipe.add(liquid_rank, 'liquid_rank')
    
    momentum_rank=momentum.rank(mask=top_500, ascending=False)
    pipe.add(momentum_rank, 'momentum_rank')
    
    # Take the average of the two factor rankings, add this to the pipeline
    combo_raw =  daily_rank
    pipe.add(combo_raw, 'combo_raw') 
    
    # Rank the combo_raw and add that to the pipeline
    pipe.add(combo_raw.rank(mask=top_500), 'combo_rank')
    
    sma_200 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=200)
    dollar_volume = AvgDailyDollarVolumeTraded()
    pipe.set_screen((sma_200 > 5) & (top_500) & (dollar_volume > 10**7))

    # Schedule my rebalance function
    schedule_function(func=rebalance, 
                      date_rule=date_rules.every_day(), 
                      time_rule=time_rules.market_close())
    schedule_function(exit, date_rules.every_day(),
                      time_rules.market_open())
    schedule_function( func=record_vars,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close())
                     
    
    # set my leverage
    context.long_leverage = 1
    context.short_leverage = 0
            
def before_trading_start(context, data):
    # Call pipelive_output to get the output
    context.output = pipeline_output('ranked_2000')
      
    # Narrow down the securities to only the top 200 & update my universe
    context.long_list = context.output.sort_values(by='combo_rank').iloc[:10]
    context.short_list = context.output.sort_values(by='combo_rank').iloc[-10:]   
    
def record_vars(context, data):  
    
     # Record and plot the leverage of our portfolio over time. 
    record(leverage = context.account.leverage)
    
    print "Long List"
    log.info("\n" + str(context.long_list.sort_values(by='combo_rank', ascending=True).head(100)))
    
    print "Short List" 
    log.info("\n" + str(context.short_list.sort_values(by='combo_rank', ascending=True).head(100)))

# This rebalancing is called according to our schedule_function settings.     
def rebalance(context,data):
    
    long_weight = context.long_leverage / float(len(context.long_list))
    short_weight = context.short_leverage / float(len(context.short_list))

    
    for long_stock in context.long_list.index :
        if data.can_trade(long_stock) :
            log.info("ordering longs")
            log.info("weight is %s" % (long_weight))
            order_target_percent(long_stock, long_weight)
           
        
    for short_stock in context.short_list.index:
        if data.can_trade(short_stock) :
            log.info("ordering shorts")
            log.info("weight is %s" % (short_weight))
            order_target_percent(short_stock, short_weight)
        
    for stock in context.portfolio.positions.iterkeys() :
        if stock not in context.long_list.index and stock not in context.short_list.index:
            order_target(stock, 0)
            
def exit(context,data):
      for context in context.portfolio.positions:
        order_target_percent(context,0)