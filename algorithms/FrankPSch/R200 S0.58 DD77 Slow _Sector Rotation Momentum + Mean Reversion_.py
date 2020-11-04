import numpy as np
import scipy
import pandas as pd

from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline, CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume, Returns
from quantopian.pipeline.filters.morningstar import Q500US, Q1500US
from quantopian.pipeline.classifiers.morningstar import Sector
import datetime

def initialize(context):
    context.mx_lvrg  = 0
    context.cash_low = 0

    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

    context.xly = sid(19662) #Consumer Discretionary 102
    context.xlp = sid(19659) #Consumer Staples 205
    context.xle = sid(19655) #Energy 309
    context.xlf = sid(19656) #Financials 103
    #context.xlv = sid(19661) #Health Care 306
    context.xli = sid(19657) #Industrials 310
    context.xlb = sid(19654) #Materials 101
    #context.xlre = sid(49472) #Real estate 104
    context.xlk = sid(19658) #Technology  311       
    context.xlu = sid(19660) #Utilities   207
    
    context.sectors = [context.xly,
                      context.xlp,
                      context.xle,
                      context.xlf,
                     # context.xlv,
                      context.xli,
                      context.xlb,
                      #context.xlre,
                      context.xlk,
                      context.xlu
                     ]
    
    context.sector_id = [102,
                      205,
                      309,
                      103,
                      #306,
                      310,
                      101,
                      #104,
                      311,
                      207
                     ]

    
    context.stock_long = []
    context.stock_short = []
    
    schedule_function(func = liquidate, date_rule = date_rules.week_end(0), time_rule = time_rules.market_close(minutes = 1))
    
    schedule_function(func = place_order, date_rule = date_rules.week_start(0), time_rule = time_rules.market_open())
    schedule_function(count_positions, date_rules.every_day(), time_rules.market_close())
    attach_pipeline(make_pipeline(context), 'mean_reversion')
    
    set_long_only()
    context.enable_short_stock = False
    
    context.today_wd = 0
    context.today_month = 0
    
def make_pipeline(context):    
    mask = Q500US()
    #lb_13 = -Returns(window_length=13, mask=mask)
    
    weekly_return = Returns(window_length=6, mask=mask)
    pipe = Pipeline(
        columns={
            'weekly_return': weekly_return,
            'sector': Sector(),
        },
        # combined_alpha will be NaN for all stocks not in our universe,
        # but we also want to make sure that we have a sector code for everything
        # we trade.
        screen=weekly_return.notnull() & Sector().notnull(),
    )
    return pipe
    
def before_trading_start(context, data):
    #print get_datetime('US/Eastern')
    context.prev_day_wd = context.today_wd
    context.today_wd = get_datetime('US/Eastern').weekday()
    
    #context.prev_day_month = context.today_month
    #context.today_month = get_datetime('US/Eastern').month
    if context.today_wd > context.prev_day_wd:
    #if context.today_month == context.prev_day_month:
        return
    else:
        # week start
        trackSectorPerformance(context, data)
        if context.bullishSectorReturn > 0 and context.bearishSectorReturn < 0:
            getBelowAverageStocksInBestPerformingSector(context, data)

    #record(cash = context.portfolio.cash)

def trackSectorPerformance(context, data):
    #log.info("in trackSectorPerformance.")
    # Rank sectors based on weekly return
    prices = data.history(context.sectors, 'price', 7, '1d')[:-1]
    daily_ret = prices.pct_change(5)[1:].as_matrix(context.sectors)
    weekly_ret = daily_ret[4]

    
    bull = sorted(range(len(weekly_ret)), key=lambda i: weekly_ret[i])[-1]
    context.bullishSector = context.sector_id[bull] 
    context.bullishSectorReturn = weekly_ret[bull] 
    #print context.bullishSectorReturn
    
    #log.info(context.bullishSector)
    
    context.bearishSectorReturn = -99
    if context.enable_short_stock:
        bear = sorted(range(len(weekly_ret)), key=lambda i: weekly_ret[i])[0]
        context.bearishSector = context.sector_id[bear]
        context.bearishSectorReturn = weekly_ret[bear]
        #print context.bullishSectorReturn
    
    #log.info("Done trackSectorPerformance.")
    
def getBelowAverageStocksInBestPerformingSector(context, data):
    #log.info("in getBelowAverageStocksInBestPerformingSector.")
    pipeline_data = pipeline_output('mean_reversion')
    bullish_data = pipeline_data[pipeline_data['sector'] == context.bullishSector]
    
    #Filter out above average return stocks
    below_average_bullish_data = bullish_data[bullish_data['weekly_return'] < context.bullishSectorReturn]
    below_average_bullish_data = below_average_bullish_data.sort_values(by = 'weekly_return')
    context.stock_to_long = below_average_bullish_data.index[:]
    stock_to_long_returns  = below_average_bullish_data['weekly_return'] 

    #Calculate weights based on returns
    context.stock_to_long_weights = context.bullishSectorReturn - stock_to_long_returns
    context.stock_to_long_weights = context.stock_to_long_weights.div(context.stock_to_long_weights.sum())
    
    #print context.stock_to_long_weights.sum()
    if context.enable_short_stock:
        bearish_data = pipeline_data[pipeline_data['sector'] == context.bearishSector]
        
        #context.stock_to_short = bearish_data.sort_values(by = 'weekly_return').index[-3:]
        #Filter out above average return stocks
        above_average_bearish_data = bearish_data[bearish_data['weekly_return'] > context.bearishSectorReturn]
        above_average_bearish_data = above_average_bearish_data.sort_values(by = 'weekly_return')
        context.stock_to_short = above_average_bearish_data.index[:]
        stock_to_short_returns  = above_average_bearish_data['weekly_return'] 

        #Calculate weights based on returns
        context.stock_to_short_weights = stock_to_short_returns - context.bearishSectorReturn
        context.stock_to_short_weights = context.stock_to_short_weights.div(context.stock_to_short_weights.sum())
 

    #log.info("done getBelowAverageStocksInBestPerformingSector.")
    
def place_order(context, data):
    #log.info("in place order.")
    try:
        if len(context.stock_to_long)>0:
            for i in range(len(context.stock_to_long)):
                order_target_percent(context.stock_to_long[i], context.stock_to_long_weights[i])
            context.stock_long = context.stock_to_long
            context.stock_to_long = []
        
        if context.enable_short_stock:
            if len(context.stock_to_short)>0:
                for i in range(len(context.stock_to_short)):
                    order_target_percent(context.stock_to_short[i], -context.stock_to_short_weights[i])
                context.stock_short = context.stock_to_short
                context.stock_to_short = []
        track_orders(context, data)

    except KeyError as k:
        # so what exactly is update_universe doing that I'm ending up here?
        print(k)
        
def liquidate(context, data):
    #log.info("in liquidate.")
    try:
        if len(context.stock_long)>0:
            log.info("Liquidating longs.")
            for i in range(len(context.stock_long)):
                order_target_percent(context.stock_long[i], 0)
           
        if context.enable_short_stock:
            if len(context.stock_short)>0:
                for i in range(len(context.stock_short)):
                    order_target_percent(context.stock_short[i], 0)

        track_orders(context, data)

    except KeyError as k:
        # so what exactly is update_universe doing that I'm ending up here?
        print(k)
    

                
def count_positions(context,data):
        
    longs = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
    leverage=context.account.leverage
    # asset=context.portfolio.portfolio_value
    # record(asset=asset)   
    record(longs=longs)
    record(leverage=leverage)  
    
def handle_data(context, data):
    track_orders(context, data)

    if context.account.leverage > context.mx_lvrg:
        context.mx_lvrg = context.account.leverage
        record(mx_lvrg = context.mx_lvrg)    # Record maximum leverage encountered

    if context.portfolio.cash < context.cash_low:
        context.cash_low = context.portfolio.cash
        record(cash_low = context.cash_low)    # Record lowest cash encountered

def track_orders(context, data):  # Log orders created, filled, unfilled or canceled.
    '''      https://www.quantopian.com/posts/track-orders
    Status:
       0 - Unfilled
       1 - Filled (can be partial)
       2 - Canceled
    '''
    c = context
    log_cash = 1    # Show cash values in logging window or not.
    log_ids  = 0    # Include order id's in logging window or not.
    log_unfilled = 1

    ''' Start and stop date options ...
    To not overwhelm the logging window, start/stop dates can be entered
      either below or in initialize() if you move to there for better efficiency.
    Example:
        c.dates  = {
            'active': 0,
            'start' : ['2007-05-07', '2010-04-26'],
            'stop'  : ['2008-02-13', '2010-11-15']
        }
    '''
    if 'orders' not in c:
        c.orders = {}               # Move these to initialize() for better efficiency.
        c.dates  = {
            'active': 0,
            'start' : [],           # Start dates, option
            'stop'  : []            # Stop  dates, option
        }
    from pytz import timezone       # Python only does once, makes this portable.
                                    #   Move to top of algo for better efficiency.

    # If the dates 'start' or 'stop' lists have something in them, sets them.
    if c.dates['start'] or c.dates['stop']:
        date = str(get_datetime().date())
        if   date in c.dates['start']:    # See if there's a match to start
            c.dates['active'] = 1
        elif date in c.dates['stop']:     #   ... or to stop
            c.dates['active'] = 0
    else:
        c.dates['active'] = 1  # Set to active b/c no conditions

    if c.dates['active'] == 0:
        return                 # Skip if off

    def _minute():   # To preface each line with the minute of the day.
        bar_dt = get_datetime().astimezone(timezone('US/Eastern'))
        minute = (bar_dt.hour * 60) + bar_dt.minute - 570  # (-570 = 9:31a)
        return str(minute).rjust(3)

    def _orders(to_log):    # So all logging comes from the same line number,
        log.info(to_log)    #   for vertical alignment in the logging window.

    ordrs = c.orders.copy()    # Independent copy to allow deletes
    for id in ordrs:
        o = get_order(id)
        if o.dt == get_datetime(): continue  # Same minute as order, no chance of fill yet.
        sec  = o.sid ; sym = sec.symbol
        oid  = o.id if log_ids else ''


        # hack for https://www.quantopian.com/posts/sector-rotation-momentum-plus-mean-reversion
        cash = 'cash {}  lv {}'.format(int(c.portfolio.cash), '%.2f' % c.account.leverage) if log_cash else ''



        prc  = '%.2f' % data.current(sec, 'price') if data.can_trade(sec) else 'unknwn'
        if o.filled:        # Filled at least some
            trade  = 'Bot' if o.amount > 0 else 'Sold'
            filled = '{}'.format(o.amount)
            filled_this = ''
            if o.filled == o.amount:    # complete
                if 0 < c.orders[o.id] < o.amount:
                    filled  = 'all {}/{}'.format(o.filled - c.orders[o.id], o.amount)
                else:
                    filled  = '{}'.format(o.amount)
                filled_this = 1
                del c.orders[o.id]
            else: # c.orders[o.id] is previously filled total
                filled_this    = o.filled - c.orders[o.id]  # filled this time, can be 0
                c.orders[o.id] = o.filled                   # save for increments math
                filled         = '{}/{}'.format(filled_this, o.amount)
            if filled_this:
                _orders(' {}      {} {} {} at {}   {} {}'.format(_minute(),
                    trade, filled, sym, prc, cash, oid))
        elif log_unfilled:
            canceled = 'canceled' if o.status == 2 else ''
            _orders(' {}         {} {} unfilled {} {}'.format(_minute(),
                    o.sid.symbol, o.amount, canceled, oid))
            if canceled: del c.orders[o.id]

    for oo_list in get_open_orders().values(): # Open orders list
        for o in oo_list:
            sec  = o.sid ; sym = sec.symbol
            oid  = o.id if log_ids else ''


            # hack for https://www.quantopian.com/posts/sector-rotation-momentum-plus-mean-reversion
            cash = 'cash {}  lv {}'.format(int(c.portfolio.cash), '%.2f' % c.account.leverage) if log_cash else ''



            prc  = '%.2f' % data.current(sec, 'price') if data.can_trade(sec) else 'unknwn'
            if o.status == 2:                  # Canceled
                _orders(' {}    Canceled {} {} order   {} {}'.format(_minute(),
                        trade, o.amount, sym, prc, cash, oid))
                del c.orders[o.id]
            elif o.id not in c.orders:         # New
                c.orders[o.id] = 0
                trade = 'Buy' if o.amount > 0 else 'Sell'
                if o.limit:                    # Limit order
                    _orders(' {}   {} {} {} now {} limit {}   {} {}'.format(_minute(),
                        trade, o.amount, sym, prc, o.limit, cash, oid))
                elif o.stop:                   # Stop order
                    _orders(' {}   {} {} {} now {} stop {}   {} {}'.format(_minute(),
                        trade, o.amount, sym, prc, o.stop, cash, oid))
                else:                          # Market order
                    _orders(' {}   {} {} {} at {}   {} {}'.format(_minute(),
                        trade, o.amount, sym, prc, cash, oid))