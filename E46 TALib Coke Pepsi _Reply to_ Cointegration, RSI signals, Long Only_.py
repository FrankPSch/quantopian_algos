import math
import numpy
import talib
from collections import deque

# Put any initialization logic here.  The context object will be passed to
# the other methods in your algorithm.
def initialize(context):
    set_long_only()    # Raise exception on short (negative share) attempt

    context.A = sid(4283)    #Coke
    context.Z = sid(5885)    #Pepsi
    context.stocks = [context.A, context.Z]
    context.threshold_buy = 30.0
    context.threshold_sell = 90.0
    context.allocation_limit = .5
    
    context.price_history = {}
    for stock in context.stocks:
        context.price_history[stock] = deque(maxlen=20)


def handle_data(context, data):
    for stock in context.stocks:
        context.price_history[stock].append(data[stock].price)
        # https://www.quantopian.com/quantopian2/migration#data-current
        # new: context.price_history[stock].append(data.current.price)
        if len(context.price_history[stock]) < 15:
            continue
            
        stock_rsi = math.floor(talib.RSI(numpy.array(context.price_history[stock]), timeperiod=14)[-1])
        log.info(stock_rsi)
        
        prices = numpy.array(context.price_history[stock])
        
        willR = talib.WILLR(prices, prices, prices, timeperiod=14)[-1]
        
        willR = 100 - abs(willR)
        
        record(willR=willR,stock_rsi=stock_rsi)
        
        if(stock_rsi > context.threshold_sell or willR > 95):
            order_target(stock, 0)
        elif(stock_rsi < context.threshold_buy and willR < 10):
            # buy up to the percentage allocation limit
            order_target_percent(stock, context.allocation_limit)