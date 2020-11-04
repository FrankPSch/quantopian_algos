#Algo du livre ameliore par communaute
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage, Latest
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar

import numpy as np
import pandas as pd
from scipy import stats
import talib


def _slope(ts):
    x = np.arange(len(ts))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts)
    # annualized_slope = np.power(np.exp(slope), 250)
    annualized_slope = (1 + slope)**250
    return annualized_slope * (r_value ** 2)      
        
        
class MarketCap(CustomFactor):   
    
    inputs = [USEquityPricing.close, morningstar.valuation.shares_outstanding] 
    window_length = 1
    
    def compute(self, today, assets, out, close, shares):       
        out[:] = close[-1] * shares[-1]
        

def make_pipeline(sma_window_length, market_cap_limit):
    
    pipe = Pipeline()  
    
    # Now only stocks in the top N largest companies by market cap
    market_cap = MarketCap()
    top_N_market_cap = market_cap.top(market_cap_limit)
    
    #Other filters to make sure we are getting a clean universe
    is_primary_share = morningstar.share_class_reference.is_primary_share.latest
    is_not_adr = ~morningstar.share_class_reference.is_depositary_receipt.latest
    
    #### TREND FITLER ON ##############
    #### We don't want to trade stocks that are below their 100 day moving average price.
    
    latest_price = USEquityPricing.close.latest
    sma = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=sma_window_length)
    above_sma = (latest_price > sma)
    initial_screen = (above_sma & top_N_market_cap & is_primary_share & is_not_adr)
    
    #### TREND FITLER OFF  ##############
    
    initial_screen = (top_N_market_cap & is_primary_share & is_not_adr)
    
    pipe.add(market_cap, "market_cap")
    
    pipe.set_screen(initial_screen)
    
    return pipe

def before_trading_start(context, data):
    context.selected_universe = pipeline_output('screen')
    context.assets = context.selected_universe.index
#    update_universe(context.selected_universe.index)

def initialize(context):
    
    context.market = sid(8554)
    context.market_window = 200
    context.atr_window = 20
    context.talib_window = context.atr_window + 5
    context.risk_factor = 0.03
    context.sma_window_length = 100
    context.momentum_window_length = 90
    context.market_cap_limit = 760
    context.rank_table_percentile = .3
    context.significant_position_difference = 0.1
    context.min_momentum = 0.080
    
    
    
    attach_pipeline(make_pipeline(context.sma_window_length,
                                  context.market_cap_limit), 'screen')
     
    # Schedule my rebalance function
    schedule_function(rebalance,
                      date_rules.month_start(),  
                      time_rules.market_open(hours=1))
    
    # Cancel all open orders at the end of each day.
    schedule_function(cancel_open_orders, date_rules.every_day(), time_rules.market_close())


def cancel_open_orders(context, data):
    open_orders = get_open_orders()
    for security in open_orders:
        for order in open_orders[security]:
            cancel_order(order)
    
    #record(lever=context.account.leverage,
    record(exposure=context.account.leverage)

def handle_data(context, data):    
    pass
    
def rebalance(context, data):
    
    highs = data.history(context.assets, "high", context.talib_window, "1d")
    lows = data.history(context.assets, "low", context.talib_window, "1d")
    closes = data.history(context.assets, "price", context.market_window, "1d")
    
    estimated_cash_balance = context.portfolio.cash
    slopes = np.log(closes[context.selected_universe.index].tail(context.momentum_window_length)).apply(_slope)
    print slopes.order(ascending=False).head(10)
    slopes = slopes[slopes > context.min_momentum]
    ranking_table = slopes[slopes > slopes.quantile(1 - context.rank_table_percentile)].order(ascending=False)

    # close positions that are no longer in the top of the ranking table
    positions = context.portfolio.positions
    for security in positions:
        price = data.current(security, "price")
        position_size = positions[security].amount
        if data.can_trade(security) and security not in ranking_table.index:
            order_target(security, 0, style=LimitOrder(price))
            estimated_cash_balance += price * position_size
        elif data.can_trade(security):
            new_position_size = get_position_size(context, highs[security], lows[security], closes[security],security)
            if significant_change_in_position_size(context, new_position_size, position_size):
                estimated_cost = price * (new_position_size * 1.5 - position_size)
                order_target(security, new_position_size * 1.5, style=LimitOrder(price))
                estimated_cash_balance -= estimated_cost
    
    
    # Market history is not used with the trend filter disabled
    # Removed for efficiency
    market_history = data.history(context.market, "price", context.market_window, "1d")  ##SPY##
    current_market_price = market_history[-1]
    average_market_price = market_history.mean()
    
    #Add liquidate all!!
    
    
    # Add new positions.
    if current_market_price > average_market_price:  ############ ac disabled market filter
    #if 1 > 0: # dummy
        for security in ranking_table.index:
            if data.can_trade(security) and security not in context.portfolio.positions:
                new_position_size = get_position_size(context, highs[security], lows[security], closes[security],
                                                     security)
                estimated_cost = data.current(security, "price") * new_position_size * 1.26
                if estimated_cash_balance > estimated_cost:
                    order_target(security, new_position_size * 1.26, style=LimitOrder(data.current(security, "price")))
                    estimated_cash_balance -= estimated_cost
    
     
def get_position_size(context, highs, lows, closes, security):
    try:
        average_true_range = talib.ATR(highs.ffill().dropna().tail(context.talib_window),
                                       lows.ffill().dropna().tail(context.talib_window),
                                       closes.ffill().dropna().tail(context.talib_window),
                                       context.atr_window)[-1] # [-1] gets the last value, as all talib methods are rolling calculations#
        return (context.portfolio.portfolio_value * context.risk_factor) / average_true_range
    except:
        log.warn('Insufficient history to calculate risk adjusted size for {0.symbol}'.format(security))
        return 0
        

def significant_change_in_position_size(context, new_position_size, old_position_size):
    return np.abs((new_position_size - old_position_size)  / old_position_size) > context.significant_position_difference
        
"""
Modifications 29/08/16

Original version came in at: 413,670  CAGR: 11.35%

1st mod: context.min_momentum = 0.020 instead of 0.30; requiring an increase of only 2% instead of 30%.
         Result:  474,679  CARG: 12.52%  + 61k
         Reason: didn't wait so long to be considered and providing more trade opportunities.
         
2nd mod: context.market_cap_limit = 700 instead of 500; again, providing more opportunities.         
         Result: 666,400  CAGR: 15.45%  + 252k above original.

3rd mod: context.rank_table_percentile = .3; was 0.20. Again provide more trade candidates.
         Result: 669,600  CAGR: 15.76%  + 276,930 above original.  #test Charles: 642.69% 
         algorithms/58be123fe2e745000d0a98ed/58cc8c75e5bb631c43419c2b#backtest

4th mod: context.risk_factor = 0.010; was 0.001. Will allow more positions to be taken
         Result: 1600.7%
         algorithms/58be123fe2e745000d0a98ed/58cc924c0d67f81757d4de5f#backtest
         
         Full backtest: 10/17/2002 a 03/17/2017 2011.1% (Market Filter OFF et Trend OFF) algorithms/58be123fe2e745000d0a98ed/58d1bcc3360fe31c90ce35f5#backtest
         Full backtest: 10/17/2002 a 03/17/2017 1524.2% (Market Filter OFF et Trend ON) algorithms/58be123fe2e745000d0a98ed/58d1c1c2efd9ea1b5a8546ff#backtest
         Full backtest: 10/17/2002 a 03/17/2017 2296% (Market Filter ON et Trend ON)
         
Modifications 22/03/17

Since htere is a flight to safety, raised momentary leverage to 1.4
         
         

"""