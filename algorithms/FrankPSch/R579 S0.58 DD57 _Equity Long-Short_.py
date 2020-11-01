# Long-Short Momentum
# 
# Inspired by Andreas Clenow's Stocks on the Move, but much different from the system in the book.
#
# Work in progress:
# - Fixed number of positions, to properly balance long and short
# - Resize SPY accordingly
# - Stocks with huge share prices e.g. YRCW. Won't buy and other them, but check logic... does it skip well
# - Penny stocks
#
# Still to do: 
# - Test correlation of different lookback lengths (63,126,252) - scuppered because of timeouts
# - Test TR (i.e. ATR(1)) to see if the averaging and parameter really matters
# - Earnings call avoidance?
# - Mid-cap and small-cap
#   https://www.quantopian.com/posts/simulating-s-and-p-500-russell-1000-russell-3000-in-research


import numpy as np
import pandas as pd
import time
from quantopian.pipeline import Pipeline
from quantopian.pipeline import CustomFactor
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import Latest, SimpleMovingAverage, Returns
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.algorithm import attach_pipeline, pipeline_output


UniverseSize = 500
DailyRangePerStock = 0.001 # targeting 10bp of account value per day
RebalanceThreshold = 0.25 # don't resize if the difference in exposure is less than this
Lookback = 126
Leverage = 1
Collateral = sid(8554) # Where to hold the cash we get from shorts: SPY to equitise returns, or None


# APR is ATR / Price, where ATR is SMA(20) of TR.
#
# It works round different backadjustment paradigms used by Quantopian:
#     https://www.quantopian.com/posts/stocks-on-the-move-by-andreas-clenow
# Uses a SMA(20) rather than the conventional Wilder exponential smoothing: 
#     http://www.macroption.com/average-true-range-calculator/
#
class APR(CustomFactor):
    inputs = [USEquityPricing.close,USEquityPricing.high,USEquityPricing.low]
    window_length = 21
    def compute(self, today, assets, out, close, high, low):
        hml = high - low
        hmpc = np.abs(high - np.roll(close, 1, axis=0))
        lmpc = np.abs(low - np.roll(close, 1, axis=0))
        tr = np.maximum(hml, np.maximum(hmpc, lmpc))
        atr = np.mean(tr[1:], axis=0) #skip the first one as it will be NaN
        apr = atr / close[-1]
        out[:] = apr
        
        
class AvgDailyDollarVolumeTraded(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.volume]
    #window_length = 100 # No default specified, do it in constructor
    def compute(self, today, assets, out, close_price, volume):
        dollar_volume = close_price * volume
        avg_dollar_volume = np.mean(dollar_volume, axis=0)
        out[:] = avg_dollar_volume


def initialize(context):
    
    context.spy = sid(8554)
    set_benchmark(context.spy)

    # define momentum as latest/SMA, same as to market (SPY) filter
    momentum        = (Latest(inputs=[USEquityPricing.close]) /
                       SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=Lookback)) - 1
    #mkt_cap         = Latest(inputs=[morningstar.valuation.market_cap]) #very slow
    #universe        = mkt_cap.top(UniverseSize)
    dollar_volume   = AvgDailyDollarVolumeTraded(window_length=100)
    universe        = dollar_volume.top(UniverseSize)
    momentum_rank   = momentum.rank(mask=universe, ascending=False)
    long_filter     = momentum_rank <= 0.2*UniverseSize
    short_filter    = momentum_rank > 0.8*UniverseSize
    apr             = APR()
    apr_filter      = apr > 0.005
    
    pipe = Pipeline()
    #pipe.add(momentum, 'momentum') # include for debugging
    pipe.add(momentum_rank, 'momentum_rank')
    pipe.add(apr, 'apr')
    pipe.add(long_filter, 'long')
    pipe.add(short_filter, 'short')
    
    pipe.set_screen( universe & apr_filter & (long_filter | short_filter) )
    pipe = attach_pipeline(pipe, name='equitylongshort')
    
    schedule_function(func=rebalance_positions, 
                      date_rule=date_rules.week_start(days_offset=2),
                      time_rule=time_rules.market_open(hours=2),
                      half_days=True)
    schedule_function(func=cancel_all,
                      date_rule=date_rules.week_start(days_offset=2),
                      time_rule=time_rules.market_close(),
                      half_days=True)
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    
def before_trading_start(context, data):
    context.pool = pipeline_output('equitylongshort').sort('momentum_rank')
    update_universe(context.pool.index)
    

def rebalance_positions(context, data):

    # There shouldn't be any open orders, as we sold at yesterday's close
    assert(not get_open_orders())   
    
    pool          = context.pool
    positions     = context.portfolio.positions
    account_value = context.portfolio.portfolio_value
    
    # Append current price and current share holding as column
    # (This works around pipeline backadjustment issue.)
    pool['price'] = [data[sid].price if sid in data else np.NaN for sid in pool.index]
    pool['current_shares'] = [positions[sid].amount if sid in positions else 0 for sid in pool.index]
    pool['is_current'] = pool.current_shares != 0
    pool['sign'] = pool.long*2-1 # +1 for longs, -1 for shorts
    
    # Calculate target number of shares (integer, can be rounded down to zero)
    pool['atr'] = (pool.apr * pool.price)
    pool['atr'].replace(0, np.NaN, inplace=True)
    pool['target_shares'] = (pool.sign * account_value * DailyRangePerStock / pool.atr).astype(int)
    pool['target_shares'].replace(np.NaN, 0, inplace=True)
    pool.loc[pool.price==0 | pool.price.isnull(), 'target_shares'] = 0
        
    # Save trading costs by not resizing current holdings if within tolerance
    pool.loc[abs(pool.current_shares - pool.target_shares) < 
             abs(pool.target_shares * RebalanceThreshold), 'target_shares'] = pool.current_shares
        
    # Sort so current holdings first, then highest momentum second
    longs  = pool[pool.long].copy()
    shorts = pool[pool.short].copy()   
    longs.sort(['is_current','momentum_rank'], inplace=True, ascending=[False,True])  
    shorts.sort(['is_current','momentum_rank'], inplace=True, ascending=[False,False])  

    # Only hold just enough to use the allowance
    longs.loc[(longs.target_shares*longs.price).cumsum() > Leverage * account_value,
                  'target_shares'] = 0
    shorts.loc[(shorts.target_shares*shorts.price).cumsum() < -Leverage * account_value,
                  'target_shares'] = 0
    '''
    # Fixed number of positions to give fixed risk (ignoring cross setional correlations)
    longs.loc[20:,'target_shares'] = 0
    shorts.loc[20:,'target_shares'] = 0'''
    
    record(LongValue=(sum(longs.target_shares*longs.price)))
    record(ShortValue=(sum(shorts.target_shares*shorts.price)))
    record(CashValue=(context.portfolio.cash))
        
    # Sell/cover positions that are no longer in the pool
    map(lambda sid: order_target(sid,0), [sid for sid in positions if (sid not in pool.index) & (sid != Collateral)])
    
    # Buy/short positions to target number of shares
    map(lambda sid: order_target(sid,longs.target_shares[sid]), [sid for sid in longs.index])
    map(lambda sid: order_target(sid,shorts.target_shares[sid]), [sid for sid in shorts.index])
    if Collateral: 
        order_target_percent(Collateral, Leverage) 
    
                   
# Called at market close, as there shouldn't be any order still active (they are "good till close")
def cancel_all(context, data):
    for security, orders in get_open_orders().iteritems():  
        for oo in orders: 
            log.warn("Cancelled %s order" % security.symbol)
            cancel_order(oo)


def handle_data(context, data):
    pass