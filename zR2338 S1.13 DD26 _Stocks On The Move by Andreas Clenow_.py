# Stocks On The Move by Andreas Clenow modified by Guy Fleury, Charles Pare and Vladimir Yetushenko 2017-03-28
# https://www.quantopian.com/posts/stocks-on-the-move-by-andreas-clenow#58d470074fde6b0be1277e29

from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage, Latest
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar

import numpy as np
import pandas as pd
from scipy import stats
import talib

def slope(ts): ## new version
    x = np.arange(len(ts))  
    log_ts = np.log(ts)  
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)  
    annualized_slope = (np.power(np.exp(slope), 250) - 1) * 100 
    return annualized_slope * (r_value ** 2) 

def _slope(ts):
    x = np.arange(len(ts))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts)
    annualized_slope = (1 + slope)**250 
    return annualized_slope * (r_value ** 2) 

class MarketCap(CustomFactor):   
    inputs = [USEquityPricing.close, morningstar.valuation.shares_outstanding] 
    window_length = 1
    
    def compute(self, today, assets, out, close, shares):       
        out[:] = close[-1] * shares[-1]
        

def make_pipeline(context,sma_window_length, market_cap_limit):
    pipe = Pipeline()  
    
    # Now only stocks in the top N largest companies by market cap
    market_cap = MarketCap()
    top_N_market_cap = market_cap.top(market_cap_limit)
    
    #Other filters to make sure we are getting a clean universe
    is_primary_share = morningstar.share_class_reference.is_primary_share.latest
    is_not_adr = ~morningstar.share_class_reference.is_depositary_receipt.latest
    
    #### TREND FITLER ##############
    #### We don't want to trade stocks that are below their sma_window_length(100) moving average price.
    if context.use_stock_trend_filter:
        latest_price = USEquityPricing.close.latest
        sma = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=sma_window_length)
        above_sma = (latest_price > sma)
        initial_screen = (above_sma & top_N_market_cap & is_primary_share & is_not_adr)
        log.info("Init: Stock trend filter ON")
    else: #### TREND FITLER OFF  ##############
        initial_screen = (top_N_market_cap & is_primary_share & is_not_adr)
        log.info("Init: Stock trend filter OFF")

    pipe.add(market_cap, "market_cap")
    
    pipe.set_screen(initial_screen)
    
    return pipe

def before_trading_start(context, data):
    context.selected_universe = pipeline_output('screen')
    context.assets = context.selected_universe.index

def initialize(context):
    context.market = sid(8554)
    context.market_window = 200
    context.atr_window = 20
    context.talib_window = context.atr_window + 5
    context.risk_factor = 0.01                     # 0.01 = less position, more % but more risk
    
    context.momentum_window_length = 90
    context.market_cap_limit = 750
    context.rank_table_percentile = .30
    context.significant_position_difference = 0.1
    context.min_momentum = 0.000
    context.leverage_factor = 2.0                   # 1=2154%. Guy's version is 1.4=3226%
    context.use_stock_trend_filter = 1              # either 0 = Off, 1 = On
    context.sma_window_length = 100                 # Used for the stock trend filter
    context.use_market_trend_filter = 1             # either 0 = Off, 1 = On. Filter on SPY
    context.use_average_true_range = 0              # either 0 = Off, 1 = On. Manage risk with individual stock volatility
    context.average_true_rage_multipl_factor = 1    # Change the weight of the ATR. 1327%
    
    
    attach_pipeline(make_pipeline(context, context.sma_window_length,
                                  context.market_cap_limit), 'screen')
     
    # Schedule my rebalance function
    schedule_function(rebalance,
                      date_rules.month_start(),  
                      time_rules.market_open(hours=1))
    
    # Cancel all open orders at the end of each day.
    schedule_function(cancel_open_orders, date_rules.every_day(), time_rules.market_close())

    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

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
    slopes = np.log(closes[context.selected_universe.index].tail(context.momentum_window_length)).apply(slope)
    print slopes.order(ascending=False).head(10)
    slopes = slopes[slopes > context.min_momentum]
    ranking_table = slopes[slopes > slopes.quantile(1 - context.rank_table_percentile)].order(ascending=False)
    log.info( len(ranking_table.index))
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
                estimated_cost = price * (new_position_size * context.leverage_factor - position_size)
                order_target(security, new_position_size * context.leverage_factor, style=LimitOrder(price))
                estimated_cash_balance -= estimated_cost
    
    
    # Market history is not used with the trend filter disabled
    # Removed for efficiency
    if context.use_market_trend_filter:
        market_history = data.history(context.market, "price", context.market_window, "1d")  ##SPY##
        current_market_price = market_history[-1]
        average_market_price = market_history.mean()
    else:
        average_market_price = 0
    
    if (current_market_price > average_market_price) :  #if average is 0 then jump in
        for security in ranking_table.index:
            if data.can_trade(security) and security not in context.portfolio.positions:
                new_position_size = get_position_size(context, highs[security], lows[security], closes[security],
                                                     security)
                estimated_cost = data.current(security, "price") * new_position_size * context.leverage_factor
                if estimated_cash_balance > estimated_cost:
                    order_target(security, new_position_size * context.leverage_factor, style=LimitOrder(data.current(security, "price")))
                    estimated_cash_balance -= estimated_cost
    
     
def get_position_size(context, highs, lows, closes, security):
    try:
        average_true_range = talib.ATR(highs.ffill().dropna().tail(context.talib_window),
                                       lows.ffill().dropna().tail(context.talib_window),
                                       closes.ffill().dropna().tail(context.talib_window),
                                       context.atr_window)[-1] # [-1] gets the last value, as all talib methods are rolling calculations#
        if not context.use_average_true_range: #average_true_range
            average_true_range = 1 #divide by 1 gives... same initial number
            context.average_true_rage_multipl_factor = 1
        
        return (context.portfolio.portfolio_value * context.risk_factor)  / (average_true_range * context.average_true_rage_multipl_factor) 
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

4th mod: context.risk_factor = 0.010; was 0.001. Will allow less positions to be taken
         Result: 1600.7%
         algorithms/58be123fe2e745000d0a98ed/58cc924c0d67f81757d4de5f#backtest


Guy Fleury
Modifications 22/03/17 

Since there is a flight to safety here, raised leverage to 1.3 with following results:
CAGR: 28.41%  with leveraging charges of $6.8M, final CAGR = 26.61%. Net liquidating value: $30M.

Total Returns 3586.3%   
Benchmark Returns 261.9%
Alpha 0.23
Beta 0.52
Sharpe 1.12
Sortino 1.69
Volatility 0.25
Max Drawdown -29%          
         
Modifications 22/03/17  Following Clenow's complaints, changed one number to increase the number of positions. Doing so reduced the bet size, drawdown, volatility and performance.

CAGR: 22.66% with leveraging charges of $737k, final CAGR = 23.36%. Net liquidating value: $20.6M.

Total Returns 2040.4%
Benchmark Returns 261.9%
Alpha 0.17
Beta 0.55
Sharpe 1.20
Sortino 1.74
Volatility 0.19
Max Drawdown -18.5%

2017-03-23 Charles Pare  Added variables to turn on or off the stock trend, market trend and ATR
2017-03-28 Vladimir Yevtushenko added new Clenow Momentum function  "slope" 
    
"""