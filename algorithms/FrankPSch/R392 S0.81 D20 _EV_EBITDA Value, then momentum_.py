#backtest note: No momentum
import pandas as pd
import numpy as np
import datetime
import math
from quantopian.pipeline.data import Fundamentals  
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.algorithm import attach_pipeline, pipeline_output


def initialize(context):   
#### Variables to change for your own liking #################
    #the constant for portfolio turnover rate
    context.holding_months = 1
    #number of stocks to pass through the fundamental screener
    context.num_screener = 100
    #number of stocks in portfolio at any time
    context.num_stock = 10
    #number of days to "look back" if employing momentum. ie formation
    context.formation_days = 200
    #set False if you want the highest momentum, True if you want low
    context.lowmom = False
    #################################################################
    #month counter for holding period logic.
    context.month_count = context.holding_months
    # Rebalance monthly on the first day of the month at market open
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    schedule_function(rebalance,
                      date_rule=date_rules.month_start(),
                      time_rule=time_rules.market_open())
    attach_pipeline(make_pipeline(), 'pipe')

def make_pipeline():
    
    filter_sectors = ((Fundamentals.morningstar_sector_code.latest != 103) &
                      (Fundamentals.morningstar_sector_code.latest != 207) #&
                      #(Fundamentals.morningstar_sector_code.latest != 206) &
                      #(Fundamentals.morningstar_sector_code.latest != 309) &
                      #(Fundamentals.morningstar_industry_code.latest != 20533080) &
                      #(Fundamentals.morningstar_industry_code.latest != 10217033) &
                      #(Fundamentals.morningstar_industry_group_code != 10106) &
                      #(Fundamentals.morningstar_industry_group_code != 10104)
                      )
    
    filter_market_cap = (Fundamentals.market_cap.latest > 2e9) #& (filter_sectors)
    filter_ev_to_ebitda = (Fundamentals.ev_to_ebitda.latest > 0) & (filter_market_cap)
    filter_enterprise_value = (Fundamentals.enterprise_value.latest > 0) & (filter_ev_to_ebitda)
    filter_shares_outstanding = (Fundamentals.shares_outstanding.latest.notnull()) & (filter_enterprise_value)
    filter_ev_to_ebitda_rank =  Fundamentals.ev_to_ebitda.latest.rank(mask = filter_shares_outstanding)
    filter_ev_to_ebitda_rank = filter_ev_to_ebitda_rank.bottom(100)
     
    pipe = Pipeline(columns = {'ev_to_ebitda': Fundamentals.ev_to_ebitda.latest,
                               'rank': Fundamentals.ev_to_ebitda.latest.rank(ascending = True),
                               'sector': Fundamentals.morningstar_sector_code.latest,
                              },screen = filter_ev_to_ebitda_rank
                   )
    return pipe

def rebalance(context, data):
    ############temp code #####################
    spy = symbol('SPY')
    if data[spy].price < data[spy].mavg(120):
        for stock in context.portfolio.positions:
            order_target(stock, 0)
        order_target_percent(symbol('TLT'), 1)
        context.month_count += 1
        print "moving towards TLT"
        return
    
    
    ####################################
    #This condition block is to skip every "holding_months"
    if context.month_count >= context.holding_months:
        context.month_count = 1
    else:
        context.month_count += 1
        return
    
    chosen_df = calc_return(context, data)
    
    #if context.num_stock < context.num_screener:
    chosen_df = sort_return(chosen_df, context.lowmom)
    chosen_df = chosen_df.iloc[:context.num_stock]
    
    # Cs for each stock
    weight = 0.99/len(chosen_df)
    # Exit all positions before starting new ones
    for stock in context.portfolio.positions:
        if stock not in chosen_df.index:
            order_target(stock, 0)
           
    # Rebalance all stocks to target weights
    for stock in chosen_df.index:
        if weight != 0 and data.can_trade(stock):
            order_target_percent(stock, weight)
    
def sort_return(df, lowmom):
    '''a cheap and quick way to sort columns according to index value. Sorts by descending order. Ie higher returns are first'''
    return df.sort(columns='return', ascending = lowmom)    


def calc_return(context, data):
    price_history = data.history(context.fundamentals.index,
                                 fields = 'price',
                                 bar_count=context.formation_days, 
                                 frequency="1d")
    temp = context.fundamentals.copy()
    
    for s in context.fundamentals.index:
        print(s)
        now = price_history[s].ix[-20] # CHANGE THIS TO -1
        old = price_history[s].ix[0]
        pct_change = (now - old) / old
        if np.isnan(pct_change):
            temp = temp.drop(s,0)
        else:
            temp.loc[s,'return'] = pct_change#calculate percent change
    
    context.stocks = temp.index
    return temp
    

def before_trading_start(context,data): 
    """
      Called before the start of each trading day. 
      It updates our universe with the
      securities and values found from fetch_fundamentals.
    """
    #this code prevents query every day
    if context.month_count != context.holding_months:
        return
    
    # Filter out only stocks that fits in criteria
    context.fundamentals = pipeline_output('pipe')
    context.stocks = context.fundamentals.index
    #print context.fundamentals