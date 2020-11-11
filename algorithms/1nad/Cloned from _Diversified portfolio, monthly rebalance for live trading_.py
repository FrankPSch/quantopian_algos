''' 
    This algorithm defines a target long-only diversified portfolio and rebalances 
    it at a user-specified frequency (currently set to 1ce per 28 days).
    
    Portfolio constituents & weights: (modify these by changing consituents in initialize()
    and changing weights in rebalance() method.

    VTI - 40 % - Vanguard Total Stock Market ETF
    VEA - 30 % - Vanguard FTSE Developed Market ETF
    VNQ - 15 % - Vanguard REIT ETF
    DJP - 10 % - iPath Dow Jones UBS Commodity ETF
    BND - 5  % - Vanguard total bond market ETF

    NOTE: This algo is intended to run in minute-mode simulation and is compatible 
          with LIVE TRADING.
'''
########### IMPORT THE LIBRARIES USED IN THE ALGORITHM ####################################
import datetime
import pytz
import pandas as pd
from zipline.utils.tradingcalendar import get_early_closes

########### INITIALZE() IS RUN ONCE (OR IN LIVE TRADING ONCE EACH DAY BEFORE TRADING) #####
def initialize(context):
    
    # Define the instruments in the portfolio:
    context.USStockMkt    = sid(22739)
    context.IntlStockMkt  = sid(34385)
    context.RealEstate    = sid(26669)
    context.NatResources  = sid(32201)
    context.Bonds         = sid(33652)
    
    # Define the benchmark (used to get early close dates for reference).
    context.spy           = sid(8554)
    
    start_date = context.spy.security_start_date
    end_date   = context.spy.security_end_date
    
    # Initialize context variables the define rebalance logic:
    context.rebalance_date = None
    context.next_rebalance_Date = None
    context.rebalance_days = 28
    context.rebalance_window_start = 10
    context.rebalance_window_stop  = 15
    
    # Get the dates when the market closes early:
    context.early_closes = get_early_closes(start_date,end_date).date

########### HANDLE_DATA() IS RUN ONCE PER MINUTE #######################
def handle_data(context, data):
    
    # Get the current exchange time, in local timezone: 
    exchange_time = pd.Timestamp(get_datetime()).tz_convert('US/Eastern')
    
    # If it is rebalance day, rebalance:
    if  context.rebalance_date == None or exchange_time >= context.next_rebalance_date:
        
       # If EOD, cancel all open orders, maintains consistency with live trading.
       cancel_orders_EOD(context)
       
       # If it's 10am proceed with rebalancing, otherwise skip this minute of trading.
       if exchange_time.hour != 10:
        return 
       
       # Check if there are any existing open orders. has_orders() defined below.
       has_orders = has_open_orders(data,context)
        
       # If we are in rebalance window but there are open orders, wait til next minute
       if has_orders == True:
            log.info('Has open orders')
            return
        
       # If there are no open orders we can rebalance.
       elif has_orders == False:
           
           rebalance(context, data, exchange_time) 
           log.info('Rebalanced portfolio to target weights at %s' % exchange_time)

           # Update the current and next rebalance dates
           context.rebalance_date = exchange_time 
           context.next_rebalance_date = context.rebalance_date + \
            datetime.timedelta(days=context.rebalance_days)      
    else:
        return
    
########### CORE REBALANCE LOGIC #########################################
## THIS FUNCTION IS RUN ONLY AT REBALANCE (DAY/TIME) #####################
def rebalance(context,data,exchange_time):
    
    #Rebalance the portfolio to the predetermined target weights:
    #
    #  40%   - US Stock Market 
    #  30%   - Foreign Stock Market
    #  15%   - Real Estate
    #  10%   - Natural Resources
    #  5.0%  - Bonds
       
    order_target_percent(context.USStockMkt,0.40)
    order_target_percent(context.IntlStockMkt,0.30)   
    order_target_percent(context.RealEstate,0.15)
    order_target_percent(context.NatResources,0.10)
    order_target_percent(context.Bonds,0.05)

########### HELPER FUNCTIONS ##############################################
        
## IN LIVE TRADE ALL OPEN ORDERS ARE CANCELLED EOD. HANDLE EXPLICITLY WITH THIS
## HELPER FUNCTION SO THAT ALL OPEN ORDERS ARE ALSO CANCELLED AT EOD IN BACKTEST.
def cancel_orders_EOD(context):

    date = get_datetime().date()
    
    # set the closing hour, based on get_early_closes (assumes that all early closes are at 13)
    if date in context.early_closes:
        close = 13 # early closing time
    else:
        close = 16 # normal closing time
    
    loc_dt = pd.Timestamp(get_datetime()).tz_convert('US/Eastern')
    
    # if it is EOD, find open orders and cancel them, otherwise return
    if loc_dt.hour == close and loc_dt.minute == 0:
        pass
    else:
        return
    
    all_open_orders = get_open_orders()
    if all_open_orders:
        for security, oo_for_sid in all_open_orders.iteritems():
            for order_obj in oo_for_sid:
                log.info("%s: Cancelling order for %s of %s created on %s" % 
                         (get_datetime(), order_obj.amount,
                          security.symbol, order_obj.created))
                cancel_order(order_obj)    
                
## RETURNS TRUE IF THERE ARE PENDING OPEN ORDERS, OTHERWISE RETURNS FALSE
def has_open_orders(data,context):               
# Only rebalance when we have zero pending orders.
    has_orders = False
    for stk in data:
        orders = get_open_orders(stk)
        if orders:
            for oo in orders:                  
                message = 'Open order for {amount} shares in {stock}'  
                message = message.format(amount=oo.amount, stock=stk)  
            has_orders = True
    return has_orders           
    if has_orders:
        return  