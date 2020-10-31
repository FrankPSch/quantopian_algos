"""
LARGE INTRADAY SWINGS PRODUCE OVERNIGHT RETURNS?
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=298092
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline, CustomFilter
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import SimpleMovingAverage, Latest
from quantopian.pipeline.filters import Q1500US 
from quantopian.pipeline.data.psychsignal import aggregated_twitter_withretweets_stocktwits as st
import pandas as pd

def custom_pipe(context):
    pipe = Pipeline()
    # bull_1 = st.bullish_intensity.latest
    # bear_1 = st.bearish_intensity.latest
    # volume = USEquityPricing.volume.latest
    sma_scan_7 = SimpleMovingAverage(inputs = [st.total_scanned_messages], window_length=7)
    # sma_scan_6 = SimpleMovingAverage(inputs = [st.total_scanned_messages], window_length=6)
    # sma_scan_5 = SimpleMovingAverage(inputs = [st.total_scanned_messages], window_length=5)
    # sma_scan_4 = SimpleMovingAverage(inputs = [st.total_scanned_messages], window_length=4)
    # sma_scan_3 = SimpleMovingAverage(inputs = [st.total_scanned_messages], window_length=3)
    # sma_scan_2 = SimpleMovingAverage(inputs = [st.total_scanned_messages], window_length=2)
    scan_1 = st.total_scanned_messages.latest    
    # pipe.add(bull_1, 'bullish_intensity')
    # pipe.add(bear_1, 'bearish_intensity')
    # pipe.add(st.bear_scored_messages.latest, 'bear_scored_messages')
    # pipe.add(st.bull_scored_messages.latest, 'bull_scored_messages')
    pipe.add(scan_1, 'scan_1') 
    # pipe.add(sma_scan_2, 'sma_scan_2') 
    # pipe.add(sma_scan_3, 'sma_scan_3') 
    # pipe.add(sma_scan_4, 'sma_scan_4') 
    # pipe.add(sma_scan_5, 'sma_scan_5') 
    # pipe.add(sma_scan_6, 'sma_scan_6') 
    pipe.add(sma_scan_7, 'sma_scan_7') 
    # pricing = USEquityPricing.close.latest

    # Base universe set to the Q500US
    base_universe = Q1500US()

    pipe.set_screen(
        (base_universe)
        # (scan_1 < 10) 
        # (pricing < 15.00) &
        # (pricing > 5.00) &
        # (volume > 1000000) &
        # (
        #     (bull_1 > 0) | 
        #     (bear_1 > 0)
        # ) 
        # (scan_1 < sma_scan_7)
    ) 
    return pipe

def initialize(context):
    '''
    initialize schedule:
    -----------:------------------:------------
    time       : function         : description
    -----------:------------------:------------
    9:30 am et : set_open_prices  : set the open prices for the equities in your universe
    3:30 pm et : set_close_prices : set the close prices for the equities in your universe
    4:00 pm et : my_rebalance     : find the equities who've had large intraday swings and buy/sell them
    -----------:------------------:------------

    '''
    attach_pipeline(custom_pipe(context), 'custom_pipe') 
    context.open_prices = None
    context.close_prices = None
    context.len_open_prices = 0
    context.len_close_prices = 0
    context.min_swing_threshhold = .1
    # context.max_swing_threshhold = .2
    
    
    # Try to sell everything at market open
    schedule_function(sell_if_profit, date_rules.every_day(), time_rules.market_open())
    
    # Set the opening price of the day.
    schedule_function(set_open_prices, date_rules.every_day(), time_rules.market_open())
    
    # 10:30
    schedule_function(sell_if_profit, date_rules.every_day(), time_rules.market_open(hours=1))
    
    # 11:30
    schedule_function(sell_if_profit, date_rules.every_day(), time_rules.market_open(hours=2))
    
    # 12:30
    schedule_function(sell_if_profit, date_rules.every_day(), time_rules.market_open(hours=3))
    
    # 1:30
    schedule_function(sell_if_profit, date_rules.every_day(), time_rules.market_open(hours=4))
    
    # 2:30
    schedule_function(sell_if_profit, date_rules.every_day(), time_rules.market_open(hours=5))
    
    # 3:30 actually sell everything
    schedule_function(sell, date_rules.every_day(), time_rules.market_open(hours=6))
     
    # Set the near closing price of the day (30 minutes before market close).
    schedule_function(set_close_prices, date_rules.every_day(), time_rules.market_close(minutes=30))

    # Find equities with the largest intraday swing and rebalance portfolio accordingly.
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_close())
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    
    

def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = pipeline_output('custom_pipe')

    # These are the securities that we are interested in trading each day.
    context.security_list = context.output.index

    # reinit trade_df everyday
    context.trade_df = pd.DataFrame()
    
    # context.trade_df["bearish_intensity"] = context.output["bearish_intensity"]
    # context.trade_df["bullish_intensity"] = context.output["bullish_intensity"]
    context.trade_df["scan_1"] = context.output["scan_1"]
    # context.trade_df["sma_scan_2"] = context.output["sma_scan_2"]
    # context.trade_df["sma_scan_3"] = context.output["sma_scan_3"]
    # context.trade_df["sma_scan_4"] = context.output["sma_scan_4"]
    # context.trade_df["sma_scan_5"] = context.output["sma_scan_5"]
    context.trade_df["sma_scan_7"] = context.output["sma_scan_7"]
    
def my_rebalance(context,data):
    """
    Execute orders according to our schedule_function() timing.
    """
    context.trade_df["intraday_swing"] = (context.trade_df["close"]/context.trade_df["open"])-1
    
    
    longs = context.trade_df[context.trade_df['intraday_swing'] < (context.min_swing_threshhold*-1)]
    longs = longs[(longs['scan_1'] < longs['sma_scan_7'])]
    
    # longs = longs[(longs['sma_scan_2'] < longs['sma_scan_3'])]
    # longs = longs[(longs['sma_scan_3'] < longs['sma_scan_4'])]
    # longs = longs[(longs['sma_scan_4'] < longs['sma_scan_5'])]
    
    # log.info(len(longs))
    # log.info((longs))
    # log.info(("-----"))
    
    shorts = context.trade_df[context.trade_df['intraday_swing'] > context.min_swing_threshhold]
    shorts = shorts[(shorts['scan_1'] < shorts['sma_scan_7'])]
    
    # shorts = shorts[(shorts['sma_scan_2'] < shorts['sma_scan_3'])]
    # shorts = shorts[(shorts['sma_scan_3'] < shorts['sma_scan_4'])]
    # shorts = shorts[(shorts['sma_scan_4'] < shorts['sma_scan_5'])]

    # log.info(len(shorts))
    # log.info((shorts))
    # log.info(("-----"))
    
    for s in longs['equity']:
        order_target_percent(s, (.2/len(longs)))
    for s in shorts['equity']:
        order_target_percent(s, (.8/len(shorts))*-1)
            
# scheduled to run every market open at 9:30
def set_open_prices(context, data):
    context.open_prices = get_curr_min_price(context, data)
    lst_equity = []
    lst_open = []
    for column in context.open_prices.columns:
        lst_equity.append(column)
    for p in context.open_prices.iloc[0]:
        lst_open.append(p)
    context.len_open_prices = len(lst_open)
    context.trade_df['equity'] = lst_equity
    context.trade_df['open'] = lst_open
    
# scheduled to run every market close at 3:59
def set_close_prices(context, data):
    context.close_prices = get_curr_min_price(context, data)
    lst_close = []
    for p in context.close_prices.iloc[0]:
        lst_close.append(p)
    context.len_close_prices = len(lst_close)
    context.trade_df['close'] = lst_close
    
def get_curr_min_price(context, data):
    security_list = context.security_list.values
    r = data.history(security_list, \
        fields="price", \
        bar_count=1, \
        frequency="1m")
    return r

def sell_if_profit(context, data):
    for s in context.portfolio.positions:
        a = context.portfolio.positions[s].amount
        cb = context.portfolio.positions[s].cost_basis
        lsp = context.portfolio.positions[s].last_sale_price
        # log.info(a)
        # log.info(cb)
        # log.info(lsp)
        # log.info("---")
        if a > 0:
            if lsp > cb:
                order_target_percent(s, 0)
        elif a < 0:
            if lsp < cb:
                order_target_percent(s, 0)

def sell(context, data):
    for s in context.portfolio.positions:
        order_target_percent(s, 0)