'''
TIPS:
For recorded values see: Full Backtest/Activity/Custom Data
'''
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
import quantopian.pipeline.filters as Filters
import quantopian.pipeline.factors as Factors
import pandas as pd
#import numpy as np
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import Returns, SimpleMovingAverage
#from quantopian.pipeline.data import Fundamentals 
#from quantopian.pipeline.filters.morningstar import Q1500US

def initialize(context):
    #context.day_count = 0
    #context.daily_message = "Day {}."
    #context.SL_Manager = StopLoss_Manager(pct_init=0.1, pct_trail=0.1)  
    #schedule_function(context.SL_Manager.manage_orders, date_rules.every_day(), time_rules.market_open())
    
    attach_pipeline(pipe_definition(context), name='my_data')
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open())
    schedule_function(record_vars, date_rules.month_start(), time_rules.market_close())
    
def pipe_definition(context):
    context.stocks = symbols(
'DBA', 'DBC', 'DIA', 'EEM', 'EFA', 'EPP', 'EWA', 'EWJ', 'EWZ', 'FXF', 'FXI', 'GLD', 'IEV', 'ILF', 'IWM', 'IYR', 'MDY', 'QQQ', 'SPY', 'TLT', 'XLE', 'XLF')
    universe = Filters.StaticAssets(context.stocks)
    #universe = Fundamentals.total_revenue.latest.top(500)
    #universe = Q1500US()
    close_price = USEquityPricing.close.latest
    
    m3 =  Returns(inputs=[USEquityPricing.close], window_length=67)*100 
    m6 =  Returns(inputs=[USEquityPricing.close], window_length=137)*100
    blend = m3*0.7+m6*0.3
    sma_88 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=89, mask=universe)

    return Pipeline(
            columns = {
            'close_price' : close_price,
            'm3': m3,
            'm6': m6,
            'blend': blend,
            'sma_88': sma_88,
            },
            screen = universe,
            )
    
def before_trading_start(context, data):
    context.output = pipeline_output('my_data')
    
    #context.day_count += 1    #log.info(context.daily_message, context.day_count)
        
def rebalance(context, data):   
    log.info(context.output.query('close_price > sma_88').sort_values(by='blend',ascending=False))
    log.info('before rebalance :                    ' + ', '.join(map(lambda x: x.symbol, context.portfolio.positions)))
       
    buy_these = context.output.query('close_price > sma_88').sort_values(by='blend',ascending=False).iloc[:5].index.tolist()
    log.info('buy_these:                             ' + str(buy_these))

    STOP_LOSS_PERCENT = .02  
    price = context.portfolio.positions[security].cost_basis * (1 -STOP_LOSS_PERCENT)  
    order_target_percent(security, 0, style=StopOrder( price )) 
    
    new_order = []
    for stock in buy_these:
         if stock not in context.portfolio.positions and data.can_trade(stock):
            new_order.append(stock)
            
    WEIGHT = 1.00/ ( len(context.portfolio.positions) + len(new_order))
    log.info(WEIGHT)
    
    for stock in context.portfolio.positions:
        order_target_percent(stock, WEIGHT, )
    log.info('after portfolios.positions rebalance :' + ', '.join(map(lambda x: x.symbol, context.portfolio.positions)))
    
    
    for stock in new_order:
        if stock not in context.portfolio.positions and data.can_trade(stock):
            order_target_percent(stock, WEIGHT)
            log.info('buy :' + str(stock) + str(WEIGHT))
    
    log.info('after new_order rebalance :            ' + ', '.join(map(lambda x: x.symbol, context.portfolio.positions)) )
    
    # cpp = context.portfolio.positions
    # log.info(cpp)  
    # cpp_symbols = map(lambda x: x.symbol, cpp)
    # log.info(cpp_symbols)
    
    sell_these = context.output.query('close_price < sma_88').index.tolist()
    for stock in sell_these:
        if stock in context.portfolio.positions and data.can_trade(stock):
            order_target_percent(stock, 0)            
            log.info('sell :' + str(stock))
    
    log.info('after sell rebalance :               ' + ', '.join(map(lambda x: x.symbol, context.portfolio.positions)))
    
            #cpp = context.portfolio.positions  
            #log.info(cpp)  
            #cpp_symbols = map(lambda x: x.symbol, cpp)  
            #log.info(cpp_symbols)
    
    #context.SL_Manager.manage_orders(context, data)  
    
def record_vars(context, data): 
    record(Leverage     = context.account.leverage)
    record(AccountValue = context.portfolio.positions_value + context.portfolio.cash)
    record(
        #sma_88 = context.output.sma_88,
        #price = context.output.close_price,
        #leverage=context.account.leverage,
        #positions=len(context.portfolio.positions)
        )

class StopLoss_Manager:
    """
    Class to manage to stop-orders for any open position or open (non-stop)-order. This will be done for long- and short-positions.
    
    Parameters:  
        pct_init (optional),
        pct_trail (optional),
        (a detailed description can be found in the set_params function)
              
    Example Usage:
        context.SL = StopLoss_Manager(pct_init=0.1, pct_trail=0.1)
        context.SL.manage_orders(context, data)
    """
                
    def set_params(self, **params):
        """
        Set values of parameters:
        
        pct_init (optional float between 0 and 1):
            - After opening a new position, this value 
              is the percentage above or below price, 
              where the first stop will be place. 
        pct_trail (optional float between 0 and 1):
            - For any existing position the price of the stop 
              will be trailed by this percentage.
        """
        additionals = set(params.keys()).difference(set(self.params.keys()))
        if len(additionals)>1:
            log.warn('Got additional parameter, which will be ignored!')
            del params[additionals]
        self.params.update(params)
       
    def manage_orders(self, context, data):
        """
        This will:
            - identify any open positions and orders with no stop
            - create new stop levels
            - manage existing stop levels
            - create StopOrders with appropriate price and amount
        """        
        self._refresh_amounts(context)
                
        for sec in self.stops.index:
            cancel_order(self.stops['id'][sec])
            if self._np.isnan(self.stops['price'][sec]):
                stop = (1-self.params['pct_init'])*data.current(sec, 'close')
            else:
                o = self._np.sign(self.stops['amount'][sec])
                new_stop = (1-o*self.params['pct_trail'])*data.current(sec, 'close')
                stop = o*max(o*self.stops['price'][sec], o*new_stop)
                
            self.stops.loc[sec, 'price'] = stop           
            self.stops.loc[sec, 'id'] = order(sec, -self.stops['amount'][sec], style=StopOrder(stop))

    def __init__(self, **params):
        """
        Creatin new StopLoss-Manager object.
        """
        self._import()
        self.params = {'pct_init': 0.01, 'pct_trail': 0.03}
        self.stops = self._pd.DataFrame(columns=['amount', 'price', 'id'])        
        self.set_params(**params)        
    
    def _refresh_amounts(self, context):
        """
        Identify open positions and orders.
        """
        
        # Reset position amounts
        self.stops.loc[:, 'amount'] = 0.
        
        # Get open orders and remember amounts for any order with no defined stop.
        open_orders = get_open_orders()
        new_amounts = []
        for sec in open_orders:
            for order in open_orders[sec]:
                if order.stop is None:
                    new_amounts.append((sec, order.amount))                
            
        # Get amounts from portfolio positions.
        for sec in context.portfolio.positions:
            new_amounts.append((sec, context.portfolio.positions[sec].amount))
            
        # Sum amounts up.
        for (sec, amount) in new_amounts:
            if not sec in self.stops.index:
                self.stops.loc[sec, 'amount'] = amount
            else:
                self.stops.loc[sec, 'amount'] = +amount
            
        # Drop securities, with no position/order any more. 
        drop = self.stops['amount'] == 0.
        self.stops.drop(self.stops.index[drop], inplace=True)
        
    def _import(self):
        """
        Import of needed packages.
        """
        import numpy
        self._np = numpy
        
        import pandas
        self._pd = pandas