# https://www.quantopian.com/posts/leverage-and-long-slash-short-exposure-one-class-to-rule-them-all
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.filters.morningstar import IsPrimaryShare
from quantopian.pipeline import factors, filters, classifiers
from quantopian.pipeline.filters import  StaticAssets, Q1500US
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage, AverageDollarVolume, Returns, Latest

import pandas as pd
import numpy as np
import scipy.stats as stats
import datetime
import math

class ExposureMngr(object):
    """
    Keep track of leverage and long/short exposure
    
    One Class to rule them all, One Class to define them,
    One Class to monitor them all and in the bytecode bind them
    
    Usage:
    Define your targets at initialization: I want leverage 1.3  and 60%/40% Long/Short balance  
       context.exposure = ExposureMngr(target_leverage = 1.3,  
                                       target_long_exposure_perc = 0.60,  
                                       target_short_exposure_perc = 0.40)  
    
    update internal state (open orders and positions)
      context.exposure.update(context, data)       
    
    After update is called, you can access the following information:
    
    how much cash available for trading  
      context.exposure.get_available_cash(consider_open_orders = True)  
    get long and short available cash as two distinct values  
      context.exposure.get_available_cash_long_short(consider_open_orders = True)  
    
    same as account.leverage but this keeps track of open orders  
      context.exposure.get_current_leverage(consider_open_orders = True)                    
    
    sum of long and short positions current value  
      context.exposure.get_exposure(consider_open_orders = True)  
    get long and short position values as two distinct values  
      context.exposure.get_long_short_exposure(consider_open_orders = True)  
    get long and short exposure as percentage  
      context.exposure.get_long_short_exposure_pct(consider_open_orders = True,  consider_unused_cash = True)
    """
    def __init__(self, target_leverage = 1.0, target_long_exposure_perc = 0.50, target_short_exposure_perc = 0.50):   
        self.target_leverage            = target_leverage
        self.target_long_exposure_perc  = target_long_exposure_perc              
        self.target_short_exposure_perc = target_short_exposure_perc           
        self.short_exposure             = 0.0
        self.long_exposure              = 0.0
        self.open_order_short_exposure  = 0.0
        self.open_order_long_exposure   = 0.0
      
    def get_current_leverage(self, context, consider_open_orders = True):
        curr_cash = context.portfolio.cash - (self.short_exposure * 2)
        if consider_open_orders:
            curr_cash -= self.open_order_short_exposure
            curr_cash -= self.open_order_long_exposure
        curr_leverage = (context.portfolio.portfolio_value - curr_cash) / context.portfolio.portfolio_value
        return curr_leverage

    def get_exposure(self, context, consider_open_orders = True):
        long_exposure, short_exposure = self.get_long_short_exposure(context, consider_open_orders)
        return long_exposure + short_exposure
    
    def get_long_short_exposure(self, context, consider_open_orders = True):
        long_exposure         = self.long_exposure
        short_exposure        = self.short_exposure
        if consider_open_orders:
            long_exposure  += self.open_order_long_exposure
            short_exposure += self.open_order_short_exposure     
        return (long_exposure, short_exposure)
    
    def get_long_short_exposure_pct(self, context, consider_open_orders = True, consider_unused_cash = True):
        long_exposure, short_exposure = self.get_long_short_exposure(context, consider_open_orders)        
        total_cash = long_exposure + short_exposure
        if consider_unused_cash:
            total_cash += self.get_available_cash(context, consider_open_orders)
        long_exposure_pct   = long_exposure  / total_cash if total_cash > 0 else 0
        short_exposure_pct  = short_exposure / total_cash if total_cash > 0 else 0
        return (long_exposure_pct, short_exposure_pct)
    
    def get_available_cash(self, context, consider_open_orders = True):
        curr_cash = context.portfolio.cash - (self.short_exposure * 2)
        if consider_open_orders:
            curr_cash -= self.open_order_short_exposure
            curr_cash -= self.open_order_long_exposure            
        leverage_cash = context.portfolio.portfolio_value * (self.target_leverage - 1.0)
        return curr_cash + leverage_cash
          
    def get_available_cash_long_short(self, context, consider_open_orders = True):
        total_available_cash  = self.get_available_cash(context, consider_open_orders)
        long_exposure         = self.long_exposure
        short_exposure        = self.short_exposure
        if consider_open_orders:
            long_exposure  += self.open_order_long_exposure
            short_exposure += self.open_order_short_exposure
        current_exposure       = long_exposure + short_exposure + total_available_cash
        target_long_exposure  = current_exposure * self.target_long_exposure_perc
        target_short_exposure = current_exposure * self.target_short_exposure_perc        
        long_available_cash   = target_long_exposure  - long_exposure 
        short_available_cash  = target_short_exposure - short_exposure
        return (long_available_cash, short_available_cash)
    
    def update(self, context, data):
        #
        # calculate cash needed to complete open orders
        #
        self.open_order_short_exposure  = 0.0
        self.open_order_long_exposure   = 0.0
        for stock, orders in  get_open_orders().iteritems():
            price = data.current(stock, 'price')
            if np.isnan(price):
                continue
            amount = 0 if stock not in context.portfolio.positions else context.portfolio.positions[stock].amount
            for oo in orders:
                order_amount = oo.amount - oo.filled
                if order_amount < 0 and amount <= 0:
                    self.open_order_short_exposure += (price * -order_amount)
                elif order_amount > 0 and amount >= 0:
                    self.open_order_long_exposure  += (price * order_amount)
            
        #
        # calculate long/short positions exposure
        #
        self.short_exposure = 0.0
        self.long_exposure  = 0.0
        for stock, position in context.portfolio.positions.iteritems():  
            amount = position.amount  
            last_sale_price = position.last_sale_price  
            if amount < 0:
                self.short_exposure += (last_sale_price * -amount)
            elif amount > 0:
                self.long_exposure  += (last_sale_price * amount)


def make_pipeline(context):      
    universe = Q1500US()
    pipe = Pipeline()
    pipe.set_screen(universe)
    pipe.add(universe, "universe")
    return pipe


# Put any initialization logic here. The context object will be passed to
# the other methods in your algorithm.
def initialize(context):
    
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=0.00))
    
    set_asset_restrictions(security_lists.restrict_leveraged_etfs)    
    context.exposure = ExposureMngr(target_leverage = 2.0,
                                    target_long_exposure_perc = 0.50,
                                    target_short_exposure_perc = 0.50)
    context.max_long_sec   = 5
    context.max_short_sec  = 5
    context.trade_freq_minutes = 20 # how often we try to enter new positions
    context.holding_minutes = 90   # max time to hold positions before liquidanting them
    
    context.trades         = []
    context.last_trade     = None
    context.can_trade      = False
    
    attach_pipeline(make_pipeline(context), 'factors')   
    
    def start_trade(context, data): context.can_trade = True
    def stop_trade(context, data): context.can_trade = False
    schedule_function(stop_trade, date_rules.every_day(), time_rules.market_close(minutes=context.holding_minutes+4))

    
def before_trading_start(context, data):    
    results = pipeline_output('factors')     
    results = results.replace([np.inf, -np.inf], np.nan)
    results = results.dropna()            
    print 'Basket of stocks %d' % (len(results))
    context.output = results
    context.universe = results.index
    context.max_lvrg = 0
    context.max_pos  = 0
    context.can_trade = True

    
def handle_data(context, data):
    
    # recors some variable (max* are for backtests, the other for live trading)
    context.max_lvrg = max(context.max_lvrg, context.account.leverage)
    context.max_pos  = max(context.max_pos, len(context.portfolio.positions))
    record(leverage=context.account.leverage,
           pos=len(context.portfolio.positions),
           max_leverage=context.max_lvrg,
           max_pos=context.max_pos)

    #    
    # try to take profit or to stop loss
    #
    check_positions_for_loss_or_profit(context, data)
    
    #
    # Enter new positions
    #
    now = get_datetime('US/Eastern')
    trade_freq = datetime.timedelta(minutes=context.trade_freq_minutes)
    if (context.last_trade is None or ((now - context.last_trade) >= trade_freq)) \
       and context.can_trade:
        context.last_trade = now
        positions = enter_positions(context,data)
        if positions:
            context.trades.append( (positions, context.holding_minutes) )
    #
    # build expected positions
    #
    expected_positions = set()
    for positions, expiration in context.trades:
        expected_positions |= set(positions)

    open_orders = get_open_orders()           
    
    #
    # Exit leftover positions (just in case something is in portfolio that shouldn't be)
    #
    log.info( 'Clearing leftover positions')
    for sec, position in context.portfolio.positions.iteritems():
        if sec not in expected_positions and sec not in open_orders:
            clear_positions(context, data, sec, position, close_open=False)
    
    #
    # Close expired positions
    #
    log.info( 'Clearing expired positions:')
    to_be_deleted = []
    
    for i, item in enumerate(context.trades):
        positions, expiration = item
        if expiration <= 0:
            to_be_deleted.append(i)
            for sec in positions:
                if sec in context.portfolio.positions:
                    position = context.portfolio.positions[sec]
                    clear_positions(context, data, sec, position, close_open=True)

    context.trades = [ (pos, exp-1) for i, (pos, exp) in enumerate(context.trades) if i not in to_be_deleted]
                
       
def clear_positions(context, data, stock, position, close_open):
    if data.can_trade(stock):
        for order in get_open_orders(stock):
            if not close_open:
                return
            log.info('Cancelling %s open order' % str(stock))
            cancel_order(order)
        amount = position.amount  
        cost_basis = position.cost_basis  
        price = data.current(stock, 'price')
        profit = amount*(price-cost_basis)
        log.info('Clearing %s %d positions (profit %d, cost %f, price %f)'
                 % (str(stock), amount, profit, cost_basis, price))
        order_target_percent(stock, 0.0)
            
def enter_positions(context,data):

    pipeout = context.output.copy()
    
    prices = data.history(pipeout.index, 'price', 200, '1m')
        
    #pipeout['low']  = prices.min()
    #pipeout['high'] = prices.max()
    #pipeout['std'] = prices.std()
    pipeout['open']  = prices.iloc[0]
    pipeout['close'] = prices.iloc[-1]
          
    pipeout['returns'] = (pipeout['close'] - pipeout['open']) / pipeout['open']
    
    #
    # rank stocks so that we can select long/short ones
    #
    longs  = pipeout.sort_values(['returns']).head(context.max_long_sec)
    shorts = pipeout.sort_values(['returns']).tail(context.max_short_sec)
  
    #
    # we don't want to enter positions that we already hold
    #
    temporary_exclusions = [ sec for sec in context.portfolio.positions ]
    longs  = longs.drop(temporary_exclusions, axis=0, errors='ignore')
    shorts = shorts.drop(temporary_exclusions, axis=0, errors='ignore')
   
    #
    # calculate available cash per long and short positions
    #
    context.exposure.update(context, data)
    long_available_cash, short_available_cash = context.exposure.get_available_cash_long_short(context)
       
    ordered_secs = []
    
    #
    # Enter long positions
    #
    cash_per_sec = long_available_cash / len(longs.index)
    for sec in longs.index:
        if data.can_trade(sec) and not get_open_orders(sec):
            amount = cash_per_sec / data.current(sec, 'price')
            log.info( 'Ordering ' + str(sec) + ' x ' + str(amount))
            order(sec, amount)
            ordered_secs.append(sec)

    # Enter short positions
    cash_per_sec = short_available_cash / len(shorts.index)
    for sec in shorts.index:
        if data.can_trade(sec) and not get_open_orders(sec):
            amount = - (cash_per_sec / data.current(sec, 'price'))
            log.info( 'Ordering ' + str(sec) + ' x ' + str(amount))
            order(sec, amount)
            ordered_secs.append(sec)
            
    return ordered_secs


def check_positions_for_loss_or_profit(context, data):
    
    # Set a value to activate risk management type
    profit_take_long  = 0.01
    stop_loss_long    = None
    
    profit_take_short = 0.01
    stop_loss_short   = None
    
    # Sell our positions on longs/shorts for profit or loss
    for security, position in context.portfolio.positions.iteritems():
                   
        if data.can_trade(security):
            
            current_position = position.amount  
            cost_basis = position.cost_basis  
            price = data.current(security, 'price')
            
            # On Long & Profit
            if profit_take_long is not None and current_position > 0:
                if price >= cost_basis * (1.0+profit_take_long):  
                    clear_positions(context, data, security, position, close_open=True)
                    log.info( str(security) + ' Sold Long for Profit')  
            # On Short & Profit
            if profit_take_short is not None and current_position < 0:
                if price <= cost_basis* (1.0-profit_take_short):
                    clear_positions(context, data, security, position, close_open=True)
                    log.info( str(security) + ' Sold Short for Profit')  
            # On Long & Loss
            if stop_loss_long is not None  and current_position > 0:
                if price <= cost_basis * (1.0-stop_loss_long):  
                    clear_positions(context, data, security, position, close_open=True)
                    log.info( str(security) + ' Sold Long for Loss')  
            # On Short & Loss
            if stop_loss_short is not None  and current_position < 0:
                if price >= cost_basis * (1.0+stop_loss_short):  
                    clear_positions(context, data, security, position, close_open=True)
                    log.info( str(security) + ' Sold Short for Loss')