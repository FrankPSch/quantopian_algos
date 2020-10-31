import numpy as np
import pytz


def initialize(context):
    context.ewa = sid(14516)
    context.ewc = sid(14517)
    
    context.delta = 0.0001
    context.Vw = context.delta / (1 - context.delta) * np.eye(2)
    context.Ve = 0.001
    
    context.beta = np.zeros(2)
    context.P = np.zeros((2, 2))
    context.R = None
    
    context.pos = None
    context.day = None
    
    set_slippage(slippage.FixedSlippage(spread=0))
    set_commission(commission.PerShare(cost=0))
    

def handle_data(context, data):
    exchange_time = get_datetime().astimezone(pytz.timezone('US/Eastern'))
    
    # update Kalman filter and exectue a trade during the last 5 mins of trading each day
    if exchange_time.hour == 15 and exchange_time.minute >= 55:
        # only execute this once per day
        if context.day is not None and context.day == exchange_time.day:
            return
        context.day = exchange_time.day
        
        x = np.asarray([data.current(context.ewa, 'price'), 1.0]).reshape((1, 2))
        y = data.current(context.ewc, 'price')
    
        # update Kalman filter with latest price
        if context.R is not None:
            context.R = context.P + context.Vw
        else:
            context.R = np.zeros((2, 2))
      
        yhat = x.dot(context.beta)
    
        Q = x.dot(context.R).dot(x.T) + context.Ve
        sqrt_Q = np.sqrt(Q)
        e = y - yhat
        K = context.R.dot(x.T) / Q
        context.beta = context.beta + K.flatten() * e
        context.P = context.R - K * x.dot(context.R)
        
        record(beta=context.beta[0], alpha=context.beta[1])
        if e < 5:
            record(spread=float(e), Q_upper=float(sqrt_Q), Q_lower=float(-sqrt_Q))
    
        if context.pos is not None:
            if context.pos == 'long' and e > -sqrt_Q:
                #log.info('closing long')
                order_target(context.ewa, 0)
                order_target(context.ewc, 0)
                context.pos = None
            elif context.pos == 'short' and e < sqrt_Q:
                #log.info('closing short')
                order_target(context.ewa, 0)
                order_target(context.ewc, 0)
                context.pos = None
            
        if context.pos is None:
            if e < -sqrt_Q:
                #log.info('opening long')
                order(context.ewc, 1000)
                order(context.ewa, -1000 * context.beta[0])
                context.pos = 'long'
            elif e > sqrt_Q:
                #log.info('opening short')
                order(context.ewc, -1000)
                order(context.ewa, 1000 * context.beta[0])
                context.pos = 'short'