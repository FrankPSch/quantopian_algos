"""
This is a template algorithm to demonstrate the usage of my StopLoss-Manager. Positions will be open after a random amount of days, if an existing position was stopped.
"""

import numpy as np
 
def initialize(context):
    
    # # Initializing the StopLoss-Manager
    context.SL_Manager = StopLoss_Manager()
    
    # Managing stop-orders for open positions at market opening.
    schedule_function(context.SL_Manager.manage_orders, date_rules.every_day(), time_rules.market_open())
    
    
    # Init parameters for position entry
    context.secs = symbols('SPY')
    context.days_wait = {key: 0  for key in context.secs}
    context.params = {'days_wait_max': 10}
    
    # Rebalance every day, 1 hour after market open.
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open(hours=1))
     
    # Record tracking variables at the end of each day.
    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())
 
def rebalance(context,data):
    """
    Execute orders according to our schedule_function() timing. 
    """
   
    to_open = set(context.secs).difference(context.portfolio.positions.keys())
                
    for sec in to_open:
        if context.days_wait[sec] > 0:
            context.days_wait[sec] -= 1   
        else:
            order_value(sec, context.portfolio.cash/len(to_open))    
            context.days_wait[sec] =  context.params['days_wait_max']
        
    # Manage stop-orders after opening/changing positions.
    context.SL_Manager.manage_orders(context, data)
 
def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    for sec in context.secs:
        if sec in context.SL_Manager.stops.index:
            record('stop price - '+ sec.symbol, context.SL_Manager.stops['price'][sec])
        else:
            record('stop price - '+ sec.symbol, None)
    
class StopLoss_Manager:
    """
    Class to manage to stop-orders for any open position or open (non-stop)-order. This will be done for long- and short-positions.
    
    Parameters:  
        pct_init (optional),
        pct_trail (optional),
        (a detailed description can be found in the set_params function)
              
    Example Usage:
        context.SL = StopLoss_Manager(pct_init=0.005, pct_trail=0.03)
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
