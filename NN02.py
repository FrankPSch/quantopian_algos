'''
    An estimation of market regime Momentum or Mean_Reversion is made
    based on Sharpe ratio of returns of the last sharpe_r_window and 
    based on Hurst exponent of prices of the last hurst_r_window.

    If the Hurst Exponent is < 0.5, then the series has been reverting in the opposite direction (Reversion).
    If the Hurst is > 0.5, then the series has kept moving in the same direction (Momentum).

    The Hurst exponent indicates persistent (H > 0.5) or antipersistent (H < 0.5) behavior of a trend
        for pink noise, H = 0
        for brown noise, H = 0.5
        for Lévy stable processes and truncated Lévy processes, H < 1

    https://www.quantopian.com/posts/neural-network-that-tests-for-mean-reversion-or-momentum-trending

    Hurst
        https://www.quantopian.com/posts/hurst-exponent
        https://en.wikipedia.org/wiki/Hurst_exponent
        https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing
        https://stackoverflow.com/questions/39488806/hurst-exponent-in-python

    Lévy etc.
        http://www.stochastik.uni-freiburg.de/emeriti/rueschendorf/publications/financial/
        http://page.math.tu-berlin.de/~papapan/papers/introduction.pdf
'''
import numpy
import random
from quantopian.algorithm import calendars
from numpy.random import randn

#-------------------------------------------------------------------------------------
# StopLoss_Manager
# https://www.quantopian.com/posts/how-to-manage-stop-loss
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
        Creating new StopLoss-Manager object.
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

#-------------------------------------------------------------------------------------
# Neural Network
class FeedForwardNetwork:
    '''
        This implementation of a standard feed forward network (FFN) is short and efficient, 
        using numpy's array multiplications for fast forward and backward passes.
        Copyright 2008 - Thomas Rueckstiess
        http://rueckstiess.net/snippets/show/17a86039
        
        Other FFNs:
        http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
        https://github.com/dennybritz/nn-from-scratch
        https://databoys.github.io/Feedforward/
        
        Learning Rate:
        https://stackoverflow.com/questions/11414374/neural-network-learning-rate-and-batch-weight-update
    '''
    def __init__(self, nIn, nHidden, nOut, alpha, hw = [], ow = []):
    
        # seed the generator
        #https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html
        random.seed(123)

        # learning rate
        self.alpha = alpha
                                                  
        # number of neurons in each layer
        self.nIn = nIn
        self.nHidden = nHidden
        self.nOut = nOut
 
        # initialize weights
        if not hw == [] and not ow == []:
            # initialize weights with previous data
            self.hWeights = hw 
            self.oWeights = ow
         
        else: 
            # initialize weights randomly (+1 for bias)
            self.hWeights = numpy.random.random((self.nHidden, self.nIn+1)) 
            self.oWeights = numpy.random.random((self.nOut, self.nHidden+1))
          
        # activations of neurons (sum of inputs)
        self.hActivation = numpy.zeros((self.nHidden, 1), dtype=float)
        self.oActivation = numpy.zeros((self.nOut, 1), dtype=float)
          
        # outputs of neurons (after sigmoid function)
        self.iOutput = numpy.zeros((self.nIn+1, nOut), dtype=float)      # +1 for bias
        self.hOutput = numpy.zeros((self.nHidden+1, nOut), dtype=float)  # +1 for bias
        self.oOutput = numpy.zeros((self.nOut), dtype=float)
          
        # deltas for hidden and output layer
        self.hDelta = numpy.zeros((self.nHidden), dtype=float)
        self.oDelta = numpy.zeros((self.nOut), dtype=float)   

    def forward(self, input_node):
        # set input as output of first layer
        self.iOutput[:-1, 0] = input_node
        self.iOutput[-1:, 0] = 1.0 # bias neuron = 1.0
          
        # hidden layer
        self.hActivation = numpy.dot(self.hWeights, self.iOutput)
        self.hOutput[:-1, :] = numpy.tanh(self.hActivation)
        self.hOutput[-1:, :] = 1.0 # bias neuron = 1.0
          
        # output layer
        self.oActivation = numpy.dot(self.oWeights, self.hOutput)
        self.oOutput = numpy.tanh(self.oActivation)
      
    def getOutput(self):
        return self.oOutput

    def backward(self, teach):
        error = self.oOutput - numpy.array(teach, dtype=float) 
          
        # deltas of output neurons
        self.oDelta = (1 - numpy.tanh(self.oActivation)) * numpy.tanh(self.oActivation) * error
                  
        # deltas of hidden neurons
        self.hDelta = (1 - numpy.tanh(self.hActivation)) * numpy.tanh(self.hActivation) * numpy.dot(numpy.transpose(self.oWeights[:,:-1]), self.oDelta)
                  
        # apply weight changes
        #print self.hWeights, self.hDelta, self.iOutput.transpose()
        self.hWeights = self.hWeights - self.alpha * numpy.dot(self.hDelta, numpy.transpose(self.iOutput)) 
        self.oWeights = self.oWeights - self.alpha * numpy.dot(self.oDelta, numpy.transpose(self.hOutput))

#-------------------------------------------------------------------------------------
# Calculate Hurst and Sharpe
def hurst(price):
    '''
        Calculate Hurst Exponent
    '''
    # Create the range of lag values
    lags = range(2, hurst_tr_window)
    # Calculate the array of the variances of the lagged differences
    tau = [numpy.sqrt(numpy.std(numpy.subtract(price[lag:], price[:-lag]))) for lag in lags]
    # Use a linear fit to estimate the Hurst Exponent
    m = numpy.polyfit(numpy.log(lags), numpy.log(tau), 1)
    # Return the Hurst exponent from the polyfit output
    return m[0]*2.0

def hurst_1(price):
    '''
        Calculate Hurst Exponent
    '''
    # Initiate tau and lag vectors
    tau = []; lagvec = []
    # Step through the different lag values
    for lag in range(2, hurst_tr_window):
        # Append the lag to a lag vector
        lagvec.append(lag)
        # Calculate the price difference with lag
        price_diff = numpy.subtract(price[lag:],price[:-lag])
        # Append the variance of the lagged price difference to a tau vector
        tau.append(numpy.sqrt(numpy.std(price_diff)))
    # Use a linear fit to a double-log graph to get the power
    m = numpy.polyfit(numpy.log10(lagvec),numpy.log10(tau),1)
    # Return the Hurst exponent from the polyfit output
    hurst = m[0]*2.0
    return hurst

def hurst_r(context, data):
    '''
        Calculate Hurst Exponent based on context.past_prices
    '''
    # Check whether enough data exists
    if len(context.past_prices) < hurst_r_window: # len: return the length (the number of items) of an object
        return
    # Initiate tau and lag vectors
    tau, lagvec = [], []
    # Step through the different lag values
    for lag in range(2, hurst_r_window): #include all datasets as chosen
        # Append the lag to a lag vector
        lagvec.append(lag)
        # Calculate the price difference with lag
        price_diff = numpy.subtract(context.past_prices[lag:],context.past_prices[:-lag])
        # Append the variance of the lagged price difference to a tau vector
        tau.append(numpy.sqrt(numpy.std(price_diff)))
    # Use a linear fit to a double-log graph to get the power
    m = numpy.polyfit(numpy.log10(lagvec),numpy.log10(tau),1)
    # Return the Hurst exponent from the polyfit output
    hurst = m[0]*2.0
    return hurst

def sharpe(series):
    '''
        Calculate Sharpe ratio on a rolling basis based on series
    '''
    returns = numpy.divide(numpy.diff(series),series[:-1])
    mean = numpy.mean(returns)
    std = numpy.std(returns)
    sharpe = mean/std
    # Sharpe * sqrt(number of periods in a year)
    #sharpe = sharpe*numpy.sqrt(sharpe_periods_per_year/float(sharpe_tr_window))
    sharpe = sharpe*numpy.sqrt(sharpe_periods_per_year)
    return sharpe

def sharpe_r(context, data):
    '''
        Calculate Sharpe ratio at the end of the period based on context.past_prices
    '''
    # Checks whether enough data exists
    if len(context.past_prices) < sharpe_r_window:
        return
    returns = numpy.divide(numpy.diff(context.past_prices), context.past_prices[:-1])
    mean = numpy.mean(returns)
    std = numpy.std(returns)
    sharpe = mean/std
    # Sharpe * sqrt(number of periods in a year)
    #sharpe = sharpe*numpy.sqrt(sharpe_periods_per_year/float(sharpe_r_window))
    sharpe = sharpe*numpy.sqrt(sharpe_periods_per_year)
    return sharpe

#-------------------------------------------------------------------------------------
# Create test data
def simulate_cointegration(d, n, mu, sigma, start_point_X, start_point_Y):
    '''
        Produce the mean reverting series, as F is the delta of two cointegrated series
    '''
    # This becomes a random walk if d = 0
    X = numpy.zeros(n)
    Y = numpy.zeros(n)
    #  These are the starting points of the random walk in y
    #  Be aware that X and Y are NOT coordinates but different series
    X[0] = start_point_X
    Y[0] = start_point_Y
    for t in range(1,n):
        #  Drunk and her dog cointegration equations (Michael P. Murray)
        X[t] = X[t-1] + random.gauss(mu,sigma);
        Y[t] = d*(X[t-1] - Y[t-1]) + Y[t-1] + random.gauss(mu,sigma);
    return X,Y,X - Y

def simulate_momentum_data(n, offset, sigma):
    '''
        Produce the trending time series
    '''
    # This becomes a random walk if offset is 0
    return numpy.cumsum([random.gauss(offset,sigma) for i in range(n)])

def rnd_teach():
    '''
        Produce randomly generated time series for which its known whether they are mean reverting, trending or random        
    '''
    k = random.randint(0, 2)
    hurst_ret = 0.5
    if k == 0: # if d>0: Mean Reverting data / Hurst<0.5
                                                               
        #while(hurst_ret >= 0.5):
        d = numpy.random.random()*0.40 + 0.05 # average was 0.25
        sigma = numpy.random.random()*0.40 + 0.40 # average was 0.60
        dummy, dummy, F = simulate_cointegration(d, sharpe_tr_window, 0, sigma, 0.0, 0.0)
        #F = numpy.log(randn(sharpe_tr_window)+100)
        sharpe_ret = sharpe(F[1:])
        hurst_ret = hurst(F[1:])
        #hurst_ret_1 = hurst_1(F[1:])
        #print('F', F)
        #print('k, sharpe_ret, hurst_ret', k, sharpe_ret, hurst_ret, hurst_ret_1)
    elif k == 1: # if offset=0: Momentum data / Hurst>0.5
                                                                    
        #while(hurst_ret <= 0.5):
        offset = numpy.random.random()*0.10 + 0.15 # average was 0.10
        sigma = numpy.random.random()*0.40 + 0.70 # average was 0.90
        F = simulate_momentum_data(sharpe_tr_window, offset, sigma)
        #F = numpy.log(numpy.cumsum(randn(sharpe_tr_window)+0.1)+300)
        sharpe_ret = sharpe(F[1:])
        hurst_ret = hurst(F[1:])
        #hurst_ret_1 = hurst_1(F[1:])
        #print('F', F)
        #print('k, sharpe_ret, hurst_ret', k, sharpe_ret, hurst_ret, hurst_ret_1)
    elif k == 2: # Random walk data, since offset=0
        offset = numpy.random.random()*0.10 - 0.05 # average was 0.00
        sigma = numpy.random.random()*0.40 + 0.70 # average was 0.90
        F = simulate_momentum_data(sharpe_tr_window, 0, sigma)
        #F = numpy.log(numpy.cumsum(randn(sharpe_tr_window))+1000)
        sharpe_ret = sharpe(F[1:])
        hurst_ret = hurst(F[1:])
        #hurst_ret_1 = hurst_1(F[1:])
        #print('F', F)
        #print('k, sharpe_ret, hurst_ret', k, sharpe_ret, hurst_ret, hurst_ret_1)
    return k, sharpe_ret, hurst_ret

#-------------------------------------------------------------------------------------
# Quantopian
def gather_prices(context, data, sid):
    ''' 
        Get context.past_prices in data frame 'sharpe_r_window'
    '''
    context.past_prices.append(data.current(sid, 'price')) # add an item to the end of the list
    if len(context.past_prices) > sharpe_r_window:
        context.past_prices.pop(0) # remove an item from position '0' in the list
    return

def close_all_positions(context, data):
    ''' 
        Iterate over the keys in context.portfolio.positions, and close out each position
    '''
    for security in context.portfolio.positions:
        order_target_percent(security, 0)
        
def record_vars(context, data):
    ''' 
        Check how many long and short positions we have
    '''
    longs = shorts = 0
    for position in context.portfolio.positions.itervalues():
      if position.amount > 0:
        longs += 1
      elif position.amount < 0:
        shorts += 1
    if longs > 0: log.debug('Long positions:   %s' % longs)
    if shorts > 0: log.debug('Short positions: %s' % shorts)

def initialize(context):
    ''' 
        Initialize Quantopian
    '''
    # default volume share slippage model
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    
    # set the cost of trades $0.001 per share, with a minimum trade cost of $5
    set_commission(commission.PerShare(cost=0.001, min_trade_cost=5.0))

    #init available cash
    context.min_notional_value = 0
    context.max_notional_value = 1000000 # 1 Mio USD
    context.order_value = 10000 # 10 k USD

    #init stock
    context.security = sid(8554)
    context.periods_traded = 0
    context.past_prices = []

    #init neural network
    context.NN_alpha = NN_alpha
    context.hw = []
    context.ow = []

    context.true_count = 0
    context.untrue_count = 0
    context.uncertain_count = 0

    #init strategy
    context.momentum = False
    context.reverting = False
    context.momentum_old = False
    context.reverting_old = False

    # https://www.quantopian.com/tutorials/getting-started#lesson7

    # If calendars.US_EQUITIES is used, market open is usually 9:30AM ET and market close is usually 4PM ET, total 6 hours and 30 minutes
    # If calendars.US_FUTURES is used, market open is at 6:30AM ET and market close is at 5PM ET, 10 hours and 30 minutes
    total_minutes = 6*60 + 30 # adjust every minute available according to calendar
    for i in range(1, total_minutes):
        # Every 'n' minutes...
        if i % myhandle_minutes == 0:
            # ...start the schedule - beginning at 9:30AM on the US Equity calendar
            schedule_function(
                            myhandle_data,
                            date_rules.every_day(),
                            time_rules.market_open(minutes=i),
                            calendars.US_EQUITIES
                            )

    # Schedule handling of data - daily at market close
    schedule_function(
                    record_vars,
                    date_rules.week_end(),
                    time_rules.market_close(),
                    calendars.US_EQUITIES
                    )

    # Initialize Stop Loss Manager
    if use_stop_loss == True:
        context.SL_Manager = StopLoss_Manager(pct_init=SL_init, pct_trail=SL_trail)
        schedule_function(
                        context.SL_Manager.manage_orders,
                        date_rules.every_day(),
                        time_rules.market_open(),
                        calendars.US_EQUITIES
                        )

def handle_data(context,data):
  pass

def myhandle_data(context,data):
    ''' 
        Handle candle stick data according to schedule
    '''
    notional_value = context.portfolio.positions_value
    # Get context.past_prices up to 'sharpe_r_window' and increase len(context.past_prices)
    gather_prices(context, data, context.security)

    context.periods_traded += 1

    # Neural Network learning phase for initial calibration: train network with known data sets
    # In the first periods traded, the NN is learned, as there is not enough data yet
    if (context.periods_traded <= NN_tr_window):
        context.ffn = FeedForwardNetwork(nIn, nHidden, nOut, context.NN_alpha, context.hw, context.ow)

        count = 0
        while(count < NN_count):
            # get dataset
            regime_desc, sharpe, hurst = rnd_teach()
            # forward pass
            context.ffn.forward([sharpe, hurst])
            # get output
            f_output = context.ffn.getOutput()[0]
            # backward pass
            context.ffn.backward(regime_desc)
            
            if (f_output > 0.75 and regime_desc == 1): # Momentum data
                context.true_count +=1
            elif (f_output < 0.25 and regime_desc == 0): # Mean Reverting data
                context.true_count +=1
            elif (f_output >= 0.75 and regime_desc == 0 or f_output < 0.25 and regime_desc==1):
                context.untrue_count += 1
            else:
                context.uncertain_count +=1

            total = float(context.true_count) + float(context.untrue_count) + float(context.uncertain_count)
            if (float(int(total))/500 == int(total/500)):
                context.NN_alpha = context.NN_alpha * NN_alpha_decay
                log.debug('total:           %g' % total)
                #log.debug('true_count:      %g' % (float(context.true_count)/total))
                log.debug('untrue_count:    %g' % (float(context.untrue_count)/total))
                #log.debug('uncertain_count: %g' % (float(context.uncertain_count)/total))
                log.debug('oWeights:')
                log.debug(', '.join(map(str, context.ffn.oWeights.tolist())))

            context.hw = context.ffn.hWeights
            context.ow = context.ffn.oWeights
            count +=1

            if (context.periods_traded == NN_tr_window and count == NN_count):
                context.NN_alpha = context.NN_alpha * 0.1
                log.debug('Finishing last  NN initial calibration, total: %g' % total)
                total = float(context.true_count) + float(context.untrue_count) + float(context.uncertain_count)
                log.debug('total:           %g' % total)
                #log.debug('true_count:      %g' % (float(context.true_count)/total))
                log.debug('untrue_count:    %g' % (float(context.untrue_count)/total))
                #log.debug('uncertain_count: %g' % (float(context.uncertain_count)/total))
                log.debug('hWeights:')
                log.debug(', '.join(map(str, context.ffn.hWeights.tolist())))
                log.debug('oWeights:')
                log.debug(', '.join(map(str, context.ffn.oWeights.tolist())))

    #elif (context.periods_traded > NN_tr_window and len(context.past_prices) < sharpe_r_window):
        #log.debug('periods_traded > NN_tr_window and len(prices) < sharpe_r_window')

    # Ordering phase after initial calibration starts, if there are enough prices to calculate Sharpe and Hurst
    elif (len(context.past_prices) >= sharpe_r_window and len(context.past_prices) >= hurst_r_window):
            # Calculate moving average, excluding current period [:-1]
            mov_avg_price = data.history(context.security, 'price', mov_avg_periods + 1, mov_avg_schedule)[:-1].mean()
            #mov_avg_volume = data.history(context.security, 'volume', mov_avg_periods+1, mov_avg_schedule)[:-1].mean()
 
            h_output = hurst_r(context,data)
            # forward pass with real data
            context.ffn.forward([sharpe_r(context,data), h_output])
            # get output from real data
            f_output = context.ffn.getOutput()[0]

            # decode market regime and backward pass based on real data
            context.momentum = False
            context.reverting = False
            if (f_output > 0.75): # and h_output > 0.5
                context.momentum = True
                #context.ffn.backward(1) # FSC: backward pass - increase NN knowledge based on real data
                if context.momentum_old != context.momentum:
                    log.debug('Momentum change from %s' % context.momentum_old)
                    log.debug('Momentum change to %s' % context.momentum)
                    log.debug('Price:       %g' % data.current(context.security, 'price'))
                    log.debug('Mean:        %g' % mov_avg_price)
                    log.debug('f_output:    %s' % f_output)
                    log.debug('h_output:    %s' % h_output)
                    context.momentum_old = context.momentum
            elif (f_output < 0.25): # and h_output < 0.5
                context.reverting = True
                #context.ffn.backward(0) # FSC: backward pass - increase NN knowledge based on real data
                if context.reverting_old != context.reverting:
                    log.debug('Reverting change from %s' % context.reverting_old)
                    log.debug('Reverting change to %s' % context.reverting)
                    log.debug('Price:       %g' % data.current(context.security, 'price'))
                    log.debug('Mean:        %g' % mov_avg_price)
                    log.debug('f_output:    %s' % f_output)
                    log.debug('h_output:    %s' % h_output)
                    context.reverting_old = context.reverting

            # Ordering if price is below moving average...
            if data.current(context.security, 'price') < mov_avg_price*.99:
                # Go short if momentum regime found and enough money left
                if context.momentum and notional_value > context.min_notional_value + context.order_value:
                    if context.security not in get_open_orders() and data.can_trade(context.security):
                        log.debug('Momentum: order short %g' % data.current(context.security, 'price'))
                        order(context.security, -context.order_value)
                        # Stop Loss Manager after creating new orders  
                        if use_stop_loss: context.SL_Manager.manage_orders(context, data)

            # Ordering if price is below moving average...
            if data.current(context.security, 'high') < mov_avg_price*.99:
                # Go long if reverting regime found and enough money left
                if context.reverting and notional_value < context.max_notional_value - context.order_value:
                    if context.security not in get_open_orders() and data.can_trade(context.security):
                        log.debug('Reverting: order long %g' % data.current(context.security, 'price'))
                        order(context.security, context.order_value)
                        # Stop Loss Manager after creating new orders  
                        if use_stop_loss: context.SL_Manager.manage_orders(context, data)

            # Ordering if price is above moving average...
            elif data.current(context.security, 'price') > mov_avg_price*1.01:
                # Go long if momentum regime found and enough money left
                if context.momentum and notional_value < context.max_notional_value - context.order_value:
                    if context.security not in get_open_orders() and data.can_trade(context.security):
                        log.debug('Momentum: order long %g' % data.current(context.security, 'price'))
                        order(context.security, context.order_value)
                        # Stop Loss Manager after creating new orders  
                        if use_stop_loss: context.SL_Manager.manage_orders(context, data)

            # Ordering if price is above moving average...
            elif data.current(context.security, 'low') > mov_avg_price*1.01:
                # Go short if reverting regime found and enough money left
                if context.reverting and notional_value > context.min_notional_value + context.order_value:
                    # get all open orders
                    if context.security not in get_open_orders() and data.can_trade(context.security):
                        log.debug('Reverting: order short %g' % data.current(context.security, 'price'))      
                        order(context.security, -context.order_value)
                        # Stop Loss Manager after creating new orders  
                        if use_stop_loss: context.SL_Manager.manage_orders(context, data)

#-------------------------------------------------------------------------------------
# Set global variables

# Scheduling: myhandle_data will be called every myhandle_minutes
myhandle_minutes = 3

# Periods for moving average
#mov_avg_schedule = '1d'
#mov_avg_periods = 30 # 30 days if '1d'
mov_avg_schedule = '1m' # minutes
mov_avg_periods = int(10 * 6.5 * 60) # 10 days if '1m', 6:30 hours in US_Equities

# Sharpe will only return valid values if the length of the list is greater than sharpe_r_window
# Set up a data frame for the data length of 'context.past_prices'
sharpe_periods_per_year = int(210 * 6.5 * 20) # 210 days of 6:30 hours in US_Equities + myhandle_minutes = 3
sharpe_r_window = int(5 * 6.5 * 20) # 5 days of 6:30 hours in US_Equities + myhandle_minutes = 3

# Hurst will only return valid values if the length of the list is greater than hurst_r_window
# hurst_r_window needs to be smaller than sharpe_r_window
hurst_r_window = sharpe_r_window - 2

# To calculate Sharpe and Hurst during initial training precisely, a larger dataset than sharpe_r_window can be used
sharpe_tr_window = sharpe_r_window
hurst_tr_window = hurst_r_window

# Neural Network learning phase for initial calibration
NN_training_tot=3000 # total training cycles in initial NN training phase
NN_count=500         # training cycles per period during initial NN training phase to avoid 50 seconds runtime limit
NN_tr_window = max(1, int(NN_training_tot / NN_count)) # number of periods of initial NN training phase

# Neural Network learning rate
#NN_alpha = 0.035          # learning rate for myhandle_minutes = 30, sharpe_r_window = int(3* 6.5 * 2), NN_training_tot=6000
NN_alpha = 0.008          # learning rate for myhandle_minutes = 30, sharpe_r_window = int(3* 6.5 * 2), NN_training_tot=6000
NN_alpha_decay = 0.99     # learning rate decay every 1000 steps

# Neural Network layout for feed forward network
nIn = 2         # Input is Sharpe and Hurst
nHidden = 5     # Hidden Layer
nOut = 1        # Output is from Reversion (0) to Momentum (1)
#nParameters = (nHidden*(nIn+1)) + (nHidden+nOut)

# Stop Loss
use_stop_loss = False
# pct_init: after opening a new position, this value is the percentage above or below price, where the first stop will be place
SL_init=0.005
# pct_trail: for any existing position the price of the stop will be trailed by this percentage.
SL_trail=0.05