#Imports
import statsmodels.api as sm 
import quantopian.pipeline.data 
import numpy as np
import talib
import scipy

def initialize(context):

    set_benchmark(symbol('SPY'))
    context.start = 1
    context.stocks = [
                      sid(37514), 
                      sid(21508)
                     ]
                      
    context.XIV = sid(40516) #VELOCITYSHARES DAILY INVERSE VIX SHORT TERM ETN
    context.VXX = sid(38054) #IPATH S&P 500 VIX SHORT-TERM FUTURES ETN

    #Variables
    context.watch_list = []
    context.run_count = 0
    context.opt_pass_count = 0
    context.eps_vals = []
    context.Safety_Weight = 0.00
    context.maxBuy = 0.50

    #Schedules
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open())
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
    for i in range(25, 385, 25):   # (start, end, every i minutes)  
        schedule_function(watch, date_rules.every_day(), time_rules.market_open(minutes=i))
    
    #Set slippage,commision, and long only restriction
    set_long_only()

 
def my_rebalance(context,data):
    
    # my_assigned_weights(context, data)  
    
    n = 28
    WFV_limit= 14.5 
    vxx_prices = data.history(context.VXX, "price", n + 2, "1d")[:-1]
    vxx_lows = data.history(context.VXX, "low", n + 2, "1d")[:-1]
    vxx_highest = vxx_prices.rolling(window = n, center=False).max()
    
    WVF = ((vxx_highest - vxx_lows)/(vxx_highest)) * 100
    record(WVF=WVF[-1])
    
    xn = 28
    xiv_prices = data.history(context.XIV, "price", xn + 2, "1d")[:-1]
    xiv_lows = data.history(context.XIV, "low", xn + 2, "1d")[:-1]
    xiv_highest = xiv_prices.rolling(window = xn, center=False).max()
    
    XWVF = ((xiv_highest - xiv_lows)/(xiv_highest)) * 100
    record(XWVF=XWVF[-1])
    
    if (WVF[-1] >= WFV_limit and WVF[-2] < WFV_limit):
        BuyStock = context.XIV
        context.Safety_Weight = 0.0
        if context.XIV not in context.portfolio.positions:
            order_target_percent(BuyStock, context.maxBuy)
            message = 'BUY: {stock}'
            log.info(message.format(stock=BuyStock.symbol))
    if (WVF[-1] < WFV_limit and WVF[-2] >= WFV_limit) or (WVF[-1] <  XWVF[-1] and WVF[-2] >= XWVF[-2]):  
        if context.XIV in context.portfolio.positions:
            SellStock = context.XIV
            order_target_percent(SellStock, 0.00)
            message = 'Sell: {stock}'
            log.info(message.format(stock=SellStock.symbol))
        context.Safety_Weight = 1.00
        
def my_record_vars(context, data):    
    
    leverage = context.account.leverage
    record(leverage=leverage)

def watch(context, data):  # On schedule, process any sids in watch_list  
    
    prices = data.history(context.watch_list, 'close', 2, '1d') # Daily, couple days  
    for s in context.watch_list:  
        if s not in context.portfolio.positions:    # If sold elsewhere, drop it from watch  
            context.watch_list.remove(s)  
            continue

        # Slope of prices, minute  
        slp = slope(data.history(s, 'close', 60, '1m').dropna())    # Minutes, note dropna(), important  
        if slp < 0:     # Close if downward  
            log.info('sell {} at {}  prv {}  {}%'.format(s.symbol, data.current(s, 'price'), '%.2f' % prices[s][0], '%.0f' % (100 * data.current(s, 'price') / prices[s][0]) ))  
            order_target(s, 0)    # Close/sell  
            context.watch_list.remove(s)

    # Any new for price jump watch_list  
    prices = data.history(context.portfolio.positions.keys(), 'close', 2, '1d') # Couple days  
    for s in context.portfolio.positions:    # Add to watch_list with price jump  
        if not data.can_trade(s):   continue  
        if data.current(s, 'price') > 1.10 * prices[s][0]:   # If price has jumped upward  
            if s in context.watch_list: continue  
            log.info('{} to watch_list at {}  prv {}  {}%'.format(s.symbol, data.current(s, 'price'), '%.2f' % prices[s][0], '%.0f' % (100 * data.current(s, 'price') / prices[s][0]) ))  
            context.watch_list.append(s)

         
def slope(in_list):     # Return slope of regression line. [Make sure this list contains no nans]  
    
    return sm.OLS(in_list, sm.add_constant(range(-len(in_list) + 1, 1))).fit().params[-1]  # slope
    
def handle_data(context,data):
        
    if context.start == 1:
        context.start = 0
        BuyStock = context.XIV
        order_target_percent(BuyStock, 1.0)