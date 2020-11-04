from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage, AverageDollarVolume, Latest, RSI
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar as mstar
from quantopian.pipeline.filters.morningstar import IsPrimaryShare
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.data import morningstar

import numpy as np
import pandas as pd

class Value(CustomFactor):
    
    inputs = [morningstar.valuation_ratios.book_value_yield,
              morningstar.valuation_ratios.sales_yield,
              morningstar.valuation_ratios.fcf_yield] 
    
    window_length = 1
    
    def compute(self, today, assets, out, book_value, sales, fcf):
        value_table = pd.DataFrame(index=assets)
        value_table["book_value"] = book_value[-1]
        value_table["sales"] = sales[-1]
        value_table["fcf"] = fcf[-1]
        out[:] = value_table.rank().mean(axis=1)

class Momentum(CustomFactor):
    
    inputs = [USEquityPricing.close]
    window_length = 252
    
    def compute(self, today, assets, out, close):       
        out[:] = close[-20] / close[0]

class MessageVolume(CustomFactor):
    inputs = [stocktwits.total_scanned_messages]
    window_length = 21
    def compute(self, today, assets, out, msgs):
        out[:] = -np.nansum(msgs, axis=0)
        
def make_pipeline():
    """
    Create and return our pipeline.
    
    We break this piece of logic out into its own function to make it easier to
    test and modify in isolation.
    
    In particular, this function can be copy/pasted into research and run by itself.
    """
    pipe = Pipeline()
    
    initial_screen = filter_universe()

    factors = {
        "Message": MessageVolume(mask=initial_screen),
        "Momentum": Momentum(mask=initial_screen),
        "Value": Value(mask=initial_screen),
    }
    
    clean_factors = None
    for name, factor in factors.items():
        if not clean_factors:
            clean_factors = factor.isfinite()
        else:
            clean_factors = clean_factors & factor.isfinite()  
            
    combined_rank = None
    for name, factor in factors.items():
        if not combined_rank:
            combined_rank = factor.rank(mask=clean_factors)
        else:
            combined_rank += factor.rank(mask=clean_factors)
    pipe.add(combined_rank, 'factor')

    # Build Filters representing the top and bottom 200 stocks by our combined ranking system.
    # We'll use these as our tradeable universe each day.
    longs = combined_rank.percentile_between(80, 90)
    shorts = combined_rank.percentile_between(10, 20)
    
    pipe.set_screen(longs | shorts)
    
    pipe.add(longs, 'longs')
    pipe.add(shorts, 'shorts')
    return pipe


def initialize(context):
    context.long_leverage = 1.0
    context.short_leverage = -1.0
    context.spy = sid(8554)
    
    attach_pipeline(make_pipeline(), 'ranking_example')
    
    # Used to avoid purchasing any leveraged ETFs 
    context.dont_buys = security_lists.leveraged_etf_list.current_securities(get_datetime())
     
    # Schedule my rebalance function
    schedule_function(func=rebalance, 
                      date_rule=date_rules.month_start (days_offset=0), 
                      time_rule=time_rules.market_open(hours=0,minutes=30), 
                      half_days=True)
    
    # Schedule a function to plot leverage and position count
    schedule_function(func=record_vars, 
                      date_rule=date_rules.every_day(), 
                      time_rule=time_rules.market_close(), 
                      half_days=True)

def before_trading_start(context, data):
    # Call pipeline_output to get the output
    # Note this is a dataframe where the index is the SIDs for all 
    # securities to pass my screen and the columns are the factors which
    output = pipeline_output('ranking_example')
    ranks = output['factor']
    
    long_ranks = ranks[output['longs']].rank()
    short_ranks = ranks[output['shorts']].rank()

    context.long_weights = (long_ranks / long_ranks.sum())
    log.info("Long Weights:")
    log.info(context.long_weights)
    
    context.short_weights = (short_ranks / short_ranks.sum())
    log.info("Short Weights:")
    log.info(context.short_weights)
    
    context.active_portfolio = context.long_weights.index.union(context.short_weights.index)


def record_vars(context, data):  
    
    # Record and plot the leverage, number of positions, and expsoure of our portfolio over time. 
    record(num_positions=len(context.portfolio.positions),
           exposure=context.account.net_leverage, 
           leverage=context.account.leverage)
    

# This function is scheduled to run at the start of each month.
def rebalance(context, data):
    """
    Allocate our long/short portfolio based on the weights supplied by
    context.long_weights and context.short_weights.
    """
    # Order our longs.
    log.info("ordering longs")
    for long_stock, long_weight in context.long_weights.iterkv():
        if data.can_trade(long_stock):
            if long_stock in context.dont_buys:
                continue
            order_target_percent(long_stock, context.long_leverage * long_weight)
    
    # Order our shorts.
    log.info("ordering shorts")
    for short_stock, short_weight in context.short_weights.iterkv():
        if data.can_trade(short_stock):
            if short_stock in context.dont_buys:
                continue
            order_target_percent(short_stock, context.short_leverage * short_weight)
    
    # Sell any positions in assets that are no longer in our target portfolio.
    for security in context.portfolio.positions:
        if data.can_trade(security):  # Work around inability to sell de-listed stocks.
            if security not in context.active_portfolio:
                order_target_percent(security, 0)
       
def filter_universe():  
    """
    9 filters:
        1. common stock
        2 & 3. not limited partnership - name and database check
        4. database has fundamental data
        5. not over the counter
        6. not when issued
        7. not depository receipts
        8. primary share
        9. high dollar volume
    Check Scott's notebook for more details.
    """
    common_stock = mstar.share_class_reference.security_type.latest.eq('ST00000001')
    not_lp_name = ~mstar.company_reference.standard_name.latest.matches('.* L[\\. ]?P\.?$')
    not_lp_balance_sheet = mstar.balance_sheet.limited_partnership.latest.isnull()
    have_data = mstar.valuation.market_cap.latest.notnull()
    not_otc = ~mstar.share_class_reference.exchange_id.latest.startswith('OTC')
    not_wi = ~mstar.share_class_reference.symbol.latest.endswith('.WI')
    not_depository = ~mstar.share_class_reference.is_depositary_receipt.latest
    primary_share = IsPrimaryShare()
    
    # Combine the above filters.
    tradable_filter = (common_stock & not_lp_name & not_lp_balance_sheet &
                       have_data & not_otc & not_wi & not_depository & primary_share)
    
    high_volume_tradable = AverageDollarVolume(
            window_length=21,
            mask=tradable_filter
        ).rank(ascending=False) < 500

    mask = high_volume_tradable
    
    return mask