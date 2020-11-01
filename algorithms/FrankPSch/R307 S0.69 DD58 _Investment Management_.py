from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline import CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar

import pandas as pd
import numpy as np



class Momentum(CustomFactor):   
    
    # Take information about close prices on US stocks for the last 30 days
    inputs = [USEquityPricing.close] 
    window_length = 30
    
    # Compute Price of 10 days ago / Price of 30 days ago
  
    def compute(self, today, assets, out, close):       
        out[:] = close[-10]/close[0]
        
class Pricetobook(CustomFactor):   
    
    # Take information about price-to-book ratio from Morningstar for the last day
    inputs = [morningstar.valuation_ratios.pb_ratio] 
    window_length = 1
      
    def compute(self, today, assets, out, pb):
        table = pd.DataFrame(index=assets)
        table ["pb"] = pb[-1]        
        
        # If there is no information on P/B ratio for a stock, fill it with the maximum value, and thus we will almost through these stocks away, because later we will consider only stocks with a low P/B ratio - value stocks
        
        out[:] = table.fillna(table.max()).mean(axis=1)
        
class Pricetoearnings(CustomFactor):   
    
        # Take information about price-to-earnings ratio from Morningstar for the last day
    inputs = [morningstar.valuation_ratios.pe_ratio] 
    window_length = 1
    
 
    def compute(self, today, assets, out, pe):
        table = pd.DataFrame(index=assets)
        table ["pe"] = pe[-1]        
        
        # If there is no information on P/E ratio for a stock, fill it with the maximum value, and thus we will almost through these stocks away, because later we will consider only stocks with a low P/E ratio - they seem to be undervalued (a company earns a lot of money, but it is cheap => soon it can become more expensive)
        
        out[:] = table.fillna(table.max()).mean(axis=1)
     
   
class Roa(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [morningstar.operation_ratios.roa] 
    window_length = 1
    

    def compute(self, today, assets, out, roa):
        table = pd.DataFrame(index=assets)
        table ["roa"] = roa[-1]
        
         # If there is no information on ROE ratio for a stock, fill it with the minimum value, and thus we will almost through these stocks away, because later we will consider only stocks with a high ROE
        
        out[:] =  table.fillna(table.min()).mean(axis=1)
 
class Roe(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [morningstar.operation_ratios.roe] 
    window_length = 1
    
    
    def compute(self, today, assets, out, roe):
        table = pd.DataFrame(index=assets)
        table ["roe"] = roe[-1]
        out[:] =  table.fillna(table.min()).mean(axis=1)

# Create custom factor to calculate a market cap based on yesterday's close price. We'll use this to get the top 2000 stocks by market cap

class MarketCap(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close, morningstar.valuation.shares_outstanding] 
    window_length = 1
    
    # Compute market cap value
    def compute(self, today, assets, out, close, shares):       
        out[:] = close[-1] * shares[-1]
        

def initialize(context):
    
    # Set a commission of $0.01 per share with a minimum cost per transaction equal to $1
    
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1))
    
    # Create a table called 'ranked_2000' in which we will put all necessary information about stocks
    
    pipe = Pipeline()
    attach_pipeline(pipe, 'ranked_2000')
    
    # In this table create a column which will contain momentum for each stock
    
    momentum = Momentum()
    pipe.add(momentum, 'momentum')
    
    # Create a column which will contain P/B ratio for each stock
    
    pb = Pricetobook()
    pipe.add(pb, 'pb')
    
    # Create a column which will contain P/E ratio for each stock
    
    pe = Pricetoearnings()
    pipe.add(pe, 'pe')
    
    # Create a column which will contain ROE ratio for each stock
    
    roe = Roe()
    pipe.add(roe, 'roe')
         
    # Create and apply a filter representing the top 2000 equities by MarketCap every day
    
    mkt_cap = MarketCap()
    top_2000 = mkt_cap.top(2000)
    
    #Create a column 'pb_rank' which contains a rank of each stock based on the ranking from the the lowest P/B ratio to the highest

    pb_rank = pb.rank(mask=top_2000, ascending=True)
    pipe.add(pb_rank, 'pb_rank')
    
    #Create a column 'pe_rank' which contains a rank of each stock based on the ranking from the the lowest P/E ratio to the highest
    
    pe_rank = pe.rank(mask=top_2000, ascending=True)
    pipe.add(pe_rank, 'pe_rank')
    
    #Create a column 'roe_rank' which contains a rank of each stock based on the ranking from the highest ROE ratio to the lowest
   
    roe_rank = roe.rank(mask=top_2000, ascending=False)
    pipe.add(roe_rank, 'roe_rank')
       
    #Create a new ranking based on three different ratios
    
    combo_raw = (1*pb_rank+1*pe_rank+1*roe_rank)/3
    pipe.add(combo_raw, 'combo_raw')
    
    # Rank the stocks by combo_raw and add ranked stocks to the pipeline
    pipe.add(combo_raw.rank(mask=top_2000), 'combo_rank')
    
    #Among stocks which are ranked leave only those which price 10 days ago is higher than 30 days ago.
    pipe.set_screen(top_2000  & (momentum>1))  
     
        
        
    # Scedule rebalance function: rebalance at the begininning of each month, 30 minutes after the market is open
    schedule_function(func=rebalance, 
                      date_rule=date_rules.month_start(days_offset=0), 
                      time_rule=time_rules.market_open(hours=0,minutes=30), 
                      half_days=True)
    
    # Schedule plotting function
    schedule_function(func=record_vars,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True)
    
# On Quantopian, Leverage = gross leverage = (market value of longs + abs(market value of shorts)) / net liquidation value. In order to avoid shorts and borrowing, we set leverage at 0.90
    context.long_leverage = 0.9
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
            
def before_trading_start(context, data):
    # Call pipelive_output to get the output
    context.output = pipeline_output('ranked_2000')
      
    # Narrow down the securities to only the top 20 & update universe
    context.long_list = context.output.sort_values(['combo_rank'], ascending=True).iloc[:20]
       

def record_vars(context, data):  
    
     # Record and plot the leverage of our portfolio over time. 
    record(leverage = context.account.leverage)
    
    print "Long List"
    log.info("\n" + str(context.long_list.sort_values(['combo_rank'], ascending=True).head(3)))
    
  
    
    

# This rebalancing is called according to our schedule_function settings.     
def rebalance(context,data):
    
    #
    
    try:
        
        long_weight = context.long_leverage / float(len(context.long_list))
        
    except ZeroDivisionError:
        
        long_weight = 0
        
    # Set maximum weight per single stock in order to avoid big idiosyncratic risk
    if long_weight > 0.054 :
        long_weight = 0.05
     
    # Writing information about transaction in log
            
    for long_stock in context.long_list.index:
        log.info("ordering longs")
        log.info("weight is %s" % (long_weight))
        order_target_percent(long_stock, long_weight)
    
    # Sell stocks which we do not want to have, according to the last calculations and ranking
        
    for stock in context.portfolio.positions.iterkeys():
        if stock not in context.long_list.index:
            order_target(stock, 0)