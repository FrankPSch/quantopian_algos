####Gonna Win Dat Contest Y'all

from talib import ATR
from quantopian.algorithm import attach_pipeline, pipeline_output   
from quantopian.pipeline import Pipeline
from quantopian.pipeline import CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing  
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.filters import Q1500US,Q500US   
from quantopian.pipeline.factors import AverageDollarVolume, SimpleMovingAverage, Latest
import pandas as pd
import numpy as np 
import math
RebalanceThreshold=0.005
UniverseSize = 500

class ATH(CustomFactor):
    window_length=500
    inputs=[USEquityPricing.close]
    def compute(self, today, assets, out, price):
        out[:]=np.max(price, axis=0)
        
class MarketCap(CustomFactor):   
    inputs = [USEquityPricing.close, morningstar.valuation.shares_outstanding] 
    window_length = 1
    def compute(self, today, assets, out, close, shares):       
        out[:] = close[-1] * shares[-1]

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
                
def initialize(context):
    context.sound_tribe_sector_9=1.0
    context.target_qty=5
    context.spy = sid(8554)


    set_long_only()
    # context.rebalance_int = 1
    # context.month_count = context.rebalance_int
# here is my schedule function
    # schedule_function(buy_stocks,date_rules.every_day(), time_rules.market_open(minutes=60))
    # schedule_function(sell_signal,date_rules.every_day(), time_rules.market_open())
    # set_commission(commission.PerShare(cost=0, min_trade_cost=0))
     
    schedule_function(func=allocate_1, 
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=60),
                      half_days=True)
    schedule_function(func=allocate_2, 
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=90),
                      half_days=True)
    schedule_function(func=allocate_3, 
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=120),
                      half_days=True)
    schedule_function(func=record_vars,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=1),
                      half_days=True)
    attach_pipeline(make_pipeline(context), 'myPipe')

def make_pipeline(context):  

    
        
    
    year_high=ATH()
    apr=APR()
    price1=USEquityPricing.close.latest
     
    sma_100 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=100)
    sma_filter = (price1 > sma_100)
    
    spy=Q500US()
    momentum        = (
        Latest(inputs=[USEquityPricing.close]) /
        SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=252)
        ) - 1
    apr_filter = (apr > 0.005)
    new_high= (year_high<=price1)
    dollar_volume = AverageDollarVolume(window_length=30)  
    high_dv = dollar_volume.percentile_between(90, 100)  
   
    
    
    mkt_cap = MarketCap().top(UniverseSize)
    
    

    mo_filter = (momentum > 0)
    universe=(new_high & spy & apr_filter & mkt_cap & mo_filter & high_dv & sma_filter)
        
    mo_rank   = momentum.rank(mask=universe, ascending=False)    
    pipe=Pipeline(
        columns={#"year high": year_high ,
                 # "price" : price1,  
                "mo_rank" : mo_rank,
                  "apr" : apr,
            
                
                   },
            screen=spy
    
        )
        
    return pipe
    

def before_trading_start(context,data):  
    myPipe=pipeline_output('myPipe').dropna()


    context.longs_df = myPipe

    # Creating a list of equity objects (rather than keeping the dataframe)  
   # isn't required but some do it as preference  

    context.phish = context.longs_df.sort_values(by=["mo_rank"],ascending=True)[:10]
       
    context.longs = context.phish.index.tolist()     

def sell_signal(context, data):
    cash_freed = 0.0
    s = ""
    context.stop={}
            
    for stock in context.portfolio.positions:
            position = context.portfolio.positions[stock]
            cash_worth = position.amount * position.last_sale_price
        # anything not in the pool of allowed stocks is immediately sold
                
            cost_basis=position.cost_basis
            highs = data.history(stock, 'high', 25, '1d').ffill().bfill()
            lows = data.history(stock, 'low', 25, '1d').ffill().bfill()
            closes = data.history(stock, 'close', 25, '1d').ffill().bfill()
            #front fill the values vs bfill() which back fills the values
            
                 
            current_price  = data.current(stock, 'price')
            if np.all(np.isnan(highs)):
                continue
                
            if np.all(np.isnan(lows)):
                continue
            
            if np.all(np.isnan(closes)):
                continue
#             atr_stop=0.90
            atr_stop = ATR(highs, lows, closes, timeperiod=10)[-1]*2     
            # if the stock is in the context stop then it is the previous price
            if stock in context.stop:
                stop=context.stop[stock]
# if it is not in context.stop then we will calculate the new stop price
               
            else:
                stop=cost_basis-atr_stop

# move up the stock if the current price is still greater than the stop    
            if (current_price) >= stop:
                stop=current_price-atr_stop   
                context.stop[stock]=stop

        
# if the stop price is higher then sell the position            
            elif (current_price)<stop : 
                position = context.portfolio.positions[sid]
                cash_worth = position.amount * position.last_sale_price
        # anything not in the pool of allowed stocks is immediately sold
                s = s + "%s, " % stock.symbol
                if get_open_orders(stock):
                    continue
                if data.can_trade(stock):
                    order_target_percent(stock,0)
                    # log.info("Sell: Hit Stop " +str(stock))          
                    cash_freed = cash_freed + cash_worth
    log.info(s)
    return cash_freed


def rebalance_positions(context, data):
    account_value = context.account.equity_with_loan
    cash_freed = 0.0
    s = ""
    for stock in context.portfolio.positions:
        position = context.portfolio.positions[stock]
        current_shares = position.amount
        if (stock in context.longs):
            target_shares = min(desired_position_size_in_shares(context, data, stock),
                                current_shares)
            sid_cash_freed = (current_shares - target_shares) * position.last_sale_price
            # only rebalance if we are buying or selling more than a certain pct of
            # account value, to save on transaction costs
            thres=abs(sid_cash_freed / account_value)
            if (thres > RebalanceThreshold) & data.can_trade(stock):
                s = s + "%s (%d -> %d), " % (stock.symbol, int(current_shares), int(target_shares))
                order_target(stock, target_shares)
                cash_freed = cash_freed + sid_cash_freed
    log.info(s)
    return cash_freed

def desired_position_size_in_shares(context, data, stock):
    account_value = context.account.equity_with_loan
    target_range = .001
    estimated_apr = context.phish['apr'][stock]
    assert(estimated_apr > 0.005) # should be filtering these out with CustomFactor
    estimated_atr = estimated_apr* data.current(stock, 'price')
    return (account_value * target_range) / estimated_atr


def add_positions(context, data, cash_available):   
    s = ""
    for i in range(0,len(context.phish)):
        stock = context.longs[i]
        if ((stock not in context.portfolio.positions) & data.can_trade(stock)):
            desired_shares = desired_position_size_in_shares(context, data, stock)
            cash_req = desired_shares *  data.current(stock, 'price')
            if ((cash_req < cash_available)):
                s = s + "%s (%d shares), " % (stock.symbol, int(desired_shares))
                order_target(stock, desired_shares)
                cash_available = cash_available - cash_req
    log.info(s)   
    
    
def can_buy(context, data):
    latest = data[context.spy].close_price
    h = history(200,'1d','close_price')
    avg = h[context.spy].mean()
    return latest > avg

def allocate_1(context, data):
    log.info("Selling...")
    cash_from_sales = sell_signal(context, data)
    
def allocate_2(context, data):
    log.info("Rebalancing...")
    cash_from_rebalance = rebalance_positions(context, data)

def allocate_3(context, data):
    if can_buy(context,data):
        log.info("Buying...")
        add_positions(context, data, context.portfolio.cash)
    else :
        log.info("Don't add, mkt bad")

                 
def record_vars(context, data):
    record(PctCash=(context.portfolio.cash / context.account.equity_with_loan))
    pos_count = len([s for s in context.portfolio.positions if context.portfolio.positions[s].amount != 0])
    record(Stocks=(pos_count / 100.0)) # scale so that the other numbers don't get squished
    record(leverage=context.account.leverage)    

def cancel_all(context, data):
    sids_cancelled = set()
    open_orders = get_open_orders()
    for security, orders in open_orders.iteritems():  
        for oo in orders: 
            sids_cancelled.add(oo.sid)
            cancel_order(oo)
    n_cancelled = len(sids_cancelled)
    if (n_cancelled > 0):
        log.info("Cancelled %d orders" % n_cancelled)
    return sids_cancelled 


    