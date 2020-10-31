# https://www.quantopian.com/posts/ballistic-xiv-slash-vxx-my-best-long-xiv-slash-vxx-strategy-to-date
# "Conservative" version of the fixed algo. 20% algo, 80% treasury bonds.
#Imports
from quantopian.pipeline.data.builtin import USEquityPricing
import statsmodels.api as sm 
import quantopian.pipeline.data 
import numpy as np
import pandas as pd
import talib
import scipy

def initialize(context):
    context.VXX = symbol('VXX')
    context.XIV = symbol('XIV')
    context.BOND = symbol('IEI')
    
    #Editable parameters
    context.XIV_StopLossPct = 0.25
    
    context.VXX_StopLossPct = 0.01
    context.VXX_TakeProfitPct = 0.50
    
    context.VOLorderpct = 0.20
    context.BONDorderpct = 0.80
    
    #Used for logging
    context.StopPrice = 0
    
    context.XIV_BuyPrice = 0
    context.XIV_SellLossPrice = 0
    
    context.VXX_BuyPrice = 0
    context.VXX_SellLossPrice = 0
    context.VXX_SellProfitPrice = 0
    
    context.xiv_sell = False
    context.xiv_buy = False
    
    context.vxx_sell = False
    context.vxx_buy = False
    
    context.last_bar = False

    # On Bar Close Functions
    for x in [1,3,5]:
        schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=x, minutes=59))
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_close())
    
    # On Bar Open Functions
    schedule_function(bar_open, date_rules.every_day(), time_rules.market_open())
    for x in [2,4,6]:
        schedule_function(bar_open, date_rules.every_day(), time_rules.market_open(hours=x))
        
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
    
# "Conservative" version of the fixed algo. 20% algo, 80% treasury bonds.
#Imports
from quantopian.pipeline.data.builtin import USEquityPricing
import statsmodels.api as sm 
import quantopian.pipeline.data 
import numpy as np
import pandas as pd
import talib
import scipy

def initialize(context):
    context.VXX = symbol('VXX')
    context.XIV = symbol('XIV')
    context.BOND = symbol('IEI')
    
    #Editable parameters
    context.XIV_StopLossPct = 0.25
    
    context.VXX_StopLossPct = 0.01
    context.VXX_TakeProfitPct = 0.50
    
    context.VOLorderpct = 0.20
    context.BONDorderpct = 0.80
    
    #Used for logging
    context.StopPrice = 0
    
    context.XIV_BuyPrice = 0
    context.XIV_SellLossPrice = 0
    
    context.VXX_BuyPrice = 0
    context.VXX_SellLossPrice = 0
    context.VXX_SellProfitPrice = 0
    
    context.xiv_sell = False
    context.xiv_buy = False
    
    context.vxx_sell = False
    context.vxx_buy = False
    
    context.last_bar = False

    # On Bar Close Functions
    for x in [1,3,5]:
        schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=x, minutes=59))
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_close())
    
    # On Bar Open Functions
    schedule_function(bar_open, date_rules.every_day(), time_rules.market_open())
    for x in [2,4,6]:
        schedule_function(bar_open, date_rules.every_day(), time_rules.market_open(hours=x))
        
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
    
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) 
    set_slippage(slippage.FixedSlippage(spread=0.01))
    set_benchmark(symbol('SPY'))
    set_long_only()
    
def bar_open(context,data): 
    if context.xiv_buy is True and data.can_trade(context.XIV) and len(get_open_orders()) == 0:
        set_fixed_stop_xiv(context, data)
        order_target_percent(context.XIV, context.VOLorderpct)
        order_target_percent(context.BOND, context.BONDorderpct)
        
        context.xiv_sell = False
    context.xiv_buy = False
            
    if context.xiv_sell is True and context.portfolio.positions[context.XIV].amount > 0: 
        order_target_percent(context.XIV, 0)
        context.xiv_buy = False 
    context.xiv_sell = False
        
    if context.vxx_buy is True and context.vxx_sell is False and len(get_open_orders()) == 0: 
        if data.can_trade(context.VXX):
            set_fixed_stop_vxx(context, data)
            order_target_percent(context.VXX, context.VOLorderpct)
            order_target_percent(context.BOND, context.BONDorderpct)
            context.vxx_sell = False
    context.vxx_buy = False
            
    if context.vxx_sell is True and context.portfolio.positions[context.VXX].amount > 0: 
        order_target_percent(context.VXX, 0)
        context.vxx_buy = False 
    context.vxx_sell = False        
 
def my_rebalance(context,data):
    xiv_prices = data.history(context.XIV, "price", 1600, "1m").resample('30T',  closed='right', label='right').last().dropna()

    #convert to 2 hour:
    xiv2hr = []
    if '20:00:00+00:00' in str(xiv_prices.index[-1]) or '21:00:00+00:00' in str(xiv_prices.index[-1]):
        xiv2hr.append([xiv_prices.index[-28], xiv_prices[-28]])
        xiv2hr.append([xiv_prices.index[-27], xiv_prices[-27]])        
        xiv2hr.append([xiv_prices.index[-23], xiv_prices[-23]])
        xiv2hr.append([xiv_prices.index[-19], xiv_prices[-19]])
        xiv2hr.append([xiv_prices.index[-15], xiv_prices[-15]])
        xiv2hr.append([xiv_prices.index[-14], xiv_prices[-14]])
        xiv2hr.append([xiv_prices.index[-10], xiv_prices[-10]])
        xiv2hr.append([xiv_prices.index[-6], xiv_prices[-6]])
        xiv2hr.append([xiv_prices.index[-2], xiv_prices[-2]])
        xiv2hr.append([xiv_prices.index[-1], xiv_prices[-1]])
        context.last_bar = True
    elif '19:30:00+00:00' in str(xiv_prices.index[-1]) or '20:30:00+00:00' in str(xiv_prices.index[-1]):
        xiv2hr.append([xiv_prices.index[-31], xiv_prices[-31]])
        xiv2hr.append([xiv_prices.index[-27], xiv_prices[-27]]) 
        xiv2hr.append([xiv_prices.index[-26], xiv_prices[-26]])
        xiv2hr.append([xiv_prices.index[-22], xiv_prices[-22]])
        xiv2hr.append([xiv_prices.index[-18], xiv_prices[-18]])
        xiv2hr.append([xiv_prices.index[-14], xiv_prices[-14]]) 
        xiv2hr.append([xiv_prices.index[-13], xiv_prices[-13]])
        xiv2hr.append([xiv_prices.index[-9], xiv_prices[-9]])
        xiv2hr.append([xiv_prices.index[-5], xiv_prices[-5]])
        xiv2hr.append([xiv_prices.index[-1], xiv_prices[-1]])
        context.last_bar = False
    elif '17:30:00+00:00' in str(xiv_prices.index[-1]) or '18:30:00+00:00' in str(xiv_prices.index[-1]):
        xiv2hr.append([xiv_prices.index[-31], xiv_prices[-31]])
        xiv2hr.append([xiv_prices.index[-27], xiv_prices[-27]]) 
        xiv2hr.append([xiv_prices.index[-23], xiv_prices[-23]])
        xiv2hr.append([xiv_prices.index[-22], xiv_prices[-22]])
        xiv2hr.append([xiv_prices.index[-18], xiv_prices[-18]])
        xiv2hr.append([xiv_prices.index[-14], xiv_prices[-14]])
        xiv2hr.append([xiv_prices.index[-10], xiv_prices[-10]])
        xiv2hr.append([xiv_prices.index[-9], xiv_prices[-9]])
        xiv2hr.append([xiv_prices.index[-5], xiv_prices[-5]])
        xiv2hr.append([xiv_prices.index[-1], xiv_prices[-1]])
        context.last_bar = False
    elif '15:30:00+00:00' in str(xiv_prices.index[-1]) or '16:30:00+00:00' in str(xiv_prices.index[-1]):
        xiv2hr.append([xiv_prices.index[-31], xiv_prices[-31]])
        xiv2hr.append([xiv_prices.index[-27], xiv_prices[-27]]) 
        xiv2hr.append([xiv_prices.index[-23], xiv_prices[-23]])
        xiv2hr.append([xiv_prices.index[-19], xiv_prices[-19]])
        xiv2hr.append([xiv_prices.index[-18], xiv_prices[-18]])
        xiv2hr.append([xiv_prices.index[-14], xiv_prices[-14]])        
        xiv2hr.append([xiv_prices.index[-10], xiv_prices[-10]])
        xiv2hr.append([xiv_prices.index[-6], xiv_prices[-6]])
        xiv2hr.append([xiv_prices.index[-5], xiv_prices[-5]])
        xiv2hr.append([xiv_prices.index[-1], xiv_prices[-1]])
        context.last_bar = False
    else:
        log.error("2 HOUR CONVERSION FAILURE")
        return
    dates, vals = zip(*xiv2hr)
    s = pd.Series(vals, index=dates)
    
    rsi = talib.RSI(s,2)
    RSI5 = talib.RSI(s, 5) 

    # XIV
    # BUY RULE
    if context.xiv_buy is False and rsi[-2] < 70 and rsi[-1] >= 70 and context.portfolio.positions[context.XIV].amount == 0:
        context.xiv_buy = True

    # SELL RULE
    if rsi[-2] > 85 and rsi[-1] <= 85 and context.portfolio.positions[context.XIV].amount > 0:
        order_target_percent(context.XIV, 0)
        context.xiv_buy = False
        
    # VXX     
    if RSI5[-1] < 70:
        if rsi[-2] > 85 and rsi[-1] <= 85:
            if context.portfolio.positions[context.VXX].amount == 0:
            #if len(get_open_orders()) == 0 and context.portfolio.positions[context.VXX].amount == 0:
                context.vxx_buy = True

    if context.portfolio.positions[context.VXX].amount > 0 and len(get_open_orders()) == 0:
        if rsi[-2] < 70 and rsi[-1] >= 70:
            order_target_percent(context.VXX, 0)
        
    # panic button
    high = (data.history(context.XIV, "high", 119, "1m")).max()
    if context.last_bar:
        high = (data.history(context.XIV, "high", 29, "1m")).max()
    price = data.current(context.XIV, 'price')
    if ((high/price) - 1) > .1 and len(get_open_orders()) == 0:
        order_target_percent(context.XIV, 0)
        context.xiv_sell = False
        context.xiv_buy = False 

def set_fixed_stop_xiv(context, data):
    #Only call this once when the stock is bought
    price = data.current(context.XIV, 'price')
    context.XIV_BuyPrice = price
    context.XIV_SellLossPrice= max(context.StopPrice, price - (context.XIV_StopLossPct * price)) 
    
def set_fixed_stop_vxx(context, data):
    #Only call this once when the stock is bought
    if data.can_trade(context.VXX):
        price = data.current(context.VXX, 'price')
        context.VXX_BuyPrice = price
        context.VXX_SellLossPrice= max(context.StopPrice, price - (context.VXX_StopLossPct * price))
        context.VXX_SellProfitPrice= (price * context.VXX_TakeProfitPct) + price

def handle_data(context, data):
    if context.portfolio.positions[context.XIV].amount > 0:     
        price = data.current(context.XIV, 'price')
        # set break even
        if price - context.XIV_BuyPrice >= 1:
            if context.XIV_BuyPrice > context.XIV_SellLossPrice:
                context.XIV_SellLossPrice = context.XIV_BuyPrice
        
        #If we have a position check sell conditions
        if price <= context.XIV_SellLossPrice and len(get_open_orders()) == 0:
            order_target_percent(context.XIV,  0)
            context.xiv_sell = False
            context.xiv_buy = False

    if context.portfolio.positions[context.VXX].amount > 0: 
        price = data.current(context.VXX, 'price')    
        if price <= context.VXX_SellLossPrice and len(get_open_orders()) == 0:
            order_target_percent(context.VXX, 0)
            context.vxx_buy = False
            context.vxx_sell = False
        if price >= context.VXX_SellProfitPrice and len(get_open_orders()) == 0:
            order_target_percent(context.VXX, 0)
            context.vxx_buy = False
            context.vxx_sell = False

def my_record_vars(context, data):    
    leverage = context.account.leverage
    record(leverage=leverage)
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) 
    set_slippage(slippage.FixedSlippage(spread=0.01))
    set_benchmark(symbol('SPY'))
    set_long_only()
    
def bar_open(context,data): 
    if context.xiv_buy is True and data.can_trade(context.XIV) and len(get_open_orders()) == 0:
        set_fixed_stop_xiv(context, data)
        order_target_percent(context.XIV, context.VOLorderpct)
        order_target_percent(context.BOND, context.BONDorderpct)
        
        context.xiv_sell = False
    context.xiv_buy = False
            
    if context.xiv_sell is True and context.portfolio.positions[context.XIV].amount > 0: 
        order_target_percent(context.XIV, 0)
        context.xiv_buy = False 
    context.xiv_sell = False
        
    if context.vxx_buy is True and context.vxx_sell is False and len(get_open_orders()) == 0: 
        if data.can_trade(context.VXX):
            set_fixed_stop_vxx(context, data)
            order_target_percent(context.VXX, context.VOLorderpct)
            order_target_percent(context.BOND, context.BONDorderpct)
            context.vxx_sell = False
    context.vxx_buy = False
            
    if context.vxx_sell is True and context.portfolio.positions[context.VXX].amount > 0: 
        order_target_percent(context.VXX, 0)
        context.vxx_buy = False 
    context.vxx_sell = False        
 
def my_rebalance(context,data):
    xiv_prices = data.history(context.XIV, "price", 1600, "1m").resample('30T',  closed='right', label='right').last().dropna()

    #convert to 2 hour:
    xiv2hr = []
    if '20:00:00+00:00' in str(xiv_prices.index[-1]) or '21:00:00+00:00' in str(xiv_prices.index[-1]):
        xiv2hr.append([xiv_prices.index[-28], xiv_prices[-28]])
        xiv2hr.append([xiv_prices.index[-27], xiv_prices[-27]])        
        xiv2hr.append([xiv_prices.index[-23], xiv_prices[-23]])
        xiv2hr.append([xiv_prices.index[-19], xiv_prices[-19]])
        xiv2hr.append([xiv_prices.index[-15], xiv_prices[-15]])
        xiv2hr.append([xiv_prices.index[-14], xiv_prices[-14]])
        xiv2hr.append([xiv_prices.index[-10], xiv_prices[-10]])
        xiv2hr.append([xiv_prices.index[-6], xiv_prices[-6]])
        xiv2hr.append([xiv_prices.index[-2], xiv_prices[-2]])
        xiv2hr.append([xiv_prices.index[-1], xiv_prices[-1]])
        context.last_bar = True
    elif '19:30:00+00:00' in str(xiv_prices.index[-1]) or '20:30:00+00:00' in str(xiv_prices.index[-1]):
        xiv2hr.append([xiv_prices.index[-31], xiv_prices[-31]])
        xiv2hr.append([xiv_prices.index[-27], xiv_prices[-27]]) 
        xiv2hr.append([xiv_prices.index[-26], xiv_prices[-26]])
        xiv2hr.append([xiv_prices.index[-22], xiv_prices[-22]])
        xiv2hr.append([xiv_prices.index[-18], xiv_prices[-18]])
        xiv2hr.append([xiv_prices.index[-14], xiv_prices[-14]]) 
        xiv2hr.append([xiv_prices.index[-13], xiv_prices[-13]])
        xiv2hr.append([xiv_prices.index[-9], xiv_prices[-9]])
        xiv2hr.append([xiv_prices.index[-5], xiv_prices[-5]])
        xiv2hr.append([xiv_prices.index[-1], xiv_prices[-1]])
        context.last_bar = False
    elif '17:30:00+00:00' in str(xiv_prices.index[-1]) or '18:30:00+00:00' in str(xiv_prices.index[-1]):
        xiv2hr.append([xiv_prices.index[-31], xiv_prices[-31]])
        xiv2hr.append([xiv_prices.index[-27], xiv_prices[-27]]) 
        xiv2hr.append([xiv_prices.index[-23], xiv_prices[-23]])
        xiv2hr.append([xiv_prices.index[-22], xiv_prices[-22]])
        xiv2hr.append([xiv_prices.index[-18], xiv_prices[-18]])
        xiv2hr.append([xiv_prices.index[-14], xiv_prices[-14]])
        xiv2hr.append([xiv_prices.index[-10], xiv_prices[-10]])
        xiv2hr.append([xiv_prices.index[-9], xiv_prices[-9]])
        xiv2hr.append([xiv_prices.index[-5], xiv_prices[-5]])
        xiv2hr.append([xiv_prices.index[-1], xiv_prices[-1]])
        context.last_bar = False
    elif '15:30:00+00:00' in str(xiv_prices.index[-1]) or '16:30:00+00:00' in str(xiv_prices.index[-1]):
        xiv2hr.append([xiv_prices.index[-31], xiv_prices[-31]])
        xiv2hr.append([xiv_prices.index[-27], xiv_prices[-27]]) 
        xiv2hr.append([xiv_prices.index[-23], xiv_prices[-23]])
        xiv2hr.append([xiv_prices.index[-19], xiv_prices[-19]])
        xiv2hr.append([xiv_prices.index[-18], xiv_prices[-18]])
        xiv2hr.append([xiv_prices.index[-14], xiv_prices[-14]])        
        xiv2hr.append([xiv_prices.index[-10], xiv_prices[-10]])
        xiv2hr.append([xiv_prices.index[-6], xiv_prices[-6]])
        xiv2hr.append([xiv_prices.index[-5], xiv_prices[-5]])
        xiv2hr.append([xiv_prices.index[-1], xiv_prices[-1]])
        context.last_bar = False
    else:
        log.error("2 HOUR CONVERSION FAILURE")
        return
    dates, vals = zip(*xiv2hr)
    s = pd.Series(vals, index=dates)
    
    rsi = talib.RSI(s,2)
    RSI5 = talib.RSI(s, 5) 

    # XIV
    # BUY RULE
    if context.xiv_buy is False and rsi[-2] < 70 and rsi[-1] >= 70 and context.portfolio.positions[context.XIV].amount == 0:
        context.xiv_buy = True

    # SELL RULE
    if rsi[-2] > 85 and rsi[-1] <= 85 and context.portfolio.positions[context.XIV].amount > 0:
        order_target_percent(context.XIV, 0)
        context.xiv_buy = False
        
    # VXX     
    if RSI5[-1] < 70:
        if rsi[-2] > 85 and rsi[-1] <= 85:
            if context.portfolio.positions[context.VXX].amount == 0:
            #if len(get_open_orders()) == 0 and context.portfolio.positions[context.VXX].amount == 0:
                context.vxx_buy = True

    if context.portfolio.positions[context.VXX].amount > 0 and len(get_open_orders()) == 0:
        if rsi[-2] < 70 and rsi[-1] >= 70:
            order_target_percent(context.VXX, 0)
        
    # panic button
    high = (data.history(context.XIV, "high", 119, "1m")).max()
    if context.last_bar:
        high = (data.history(context.XIV, "high", 29, "1m")).max()
    price = data.current(context.XIV, 'price')
    if ((high/price) - 1) > .1 and len(get_open_orders()) == 0:
        order_target_percent(context.XIV, 0)
        context.xiv_sell = False
        context.xiv_buy = False 

def set_fixed_stop_xiv(context, data):
    #Only call this once when the stock is bought
    price = data.current(context.XIV, 'price')
    context.XIV_BuyPrice = price
    context.XIV_SellLossPrice= max(context.StopPrice, price - (context.XIV_StopLossPct * price)) 
    
def set_fixed_stop_vxx(context, data):
    #Only call this once when the stock is bought
    if data.can_trade(context.VXX):
        price = data.current(context.VXX, 'price')
        context.VXX_BuyPrice = price
        context.VXX_SellLossPrice= max(context.StopPrice, price - (context.VXX_StopLossPct * price))
        context.VXX_SellProfitPrice= (price * context.VXX_TakeProfitPct) + price

def handle_data(context, data):
    if context.portfolio.positions[context.XIV].amount > 0:     
        price = data.current(context.XIV, 'price')
        # set break even
        if price - context.XIV_BuyPrice >= 1:
            if context.XIV_BuyPrice > context.XIV_SellLossPrice:
                context.XIV_SellLossPrice = context.XIV_BuyPrice
        
        #If we have a position check sell conditions
        if price <= context.XIV_SellLossPrice and len(get_open_orders()) == 0:
            order_target_percent(context.XIV,  0)
            context.xiv_sell = False
            context.xiv_buy = False

    if context.portfolio.positions[context.VXX].amount > 0: 
        price = data.current(context.VXX, 'price')    
        if price <= context.VXX_SellLossPrice and len(get_open_orders()) == 0:
            order_target_percent(context.VXX, 0)
            context.vxx_buy = False
            context.vxx_sell = False
        if price >= context.VXX_SellProfitPrice and len(get_open_orders()) == 0:
            order_target_percent(context.VXX, 0)
            context.vxx_buy = False
            context.vxx_sell = False

def my_record_vars(context, data):    
    leverage = context.account.leverage
    record(leverage=leverage)